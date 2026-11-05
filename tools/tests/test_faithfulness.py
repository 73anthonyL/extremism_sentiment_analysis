"""Behavioural and adversarial tests for tools/faithfulness.py with a fake predictor."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import faithfulness as fa
from faithfulness import (
    TABLE_COLUMNS,
    FaithfulnessError,
    analyze,
    build_posts,
    check_text_free,
    delete_tokens,
    deletion_metrics,
    deletion_plan,
    load_posts,
    random_positions,
    shared_row_ids,
    top_k_positive,
    write_outputs,
    write_table,
)
from notebook_kit import MASK_TOKEN, aggregate_word_attributions

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
TRIGGER = "burn"


def _posts(n=40):
    """Synthetic posts, no dataset text. Even rows carry the trigger word."""
    rows = []
    for i in range(n):
        if i % 2 == 0:
            tokens = ["we ", "should ", f"{TRIGGER} ", "the ", "old ", "garden"]
            values = [0.01, 0.02, 0.4, -0.01, 0.0, 0.03]
        else:
            tokens = ["a ", "quiet ", "walk ", "in ", "the ", "garden"]
            values = [-0.01, -0.02, 0.01, 0.0, -0.01, -0.03]
        rows.append((f"ex_{i:06d}", tokens, np.asarray(values)))
    return rows


def _processed_and_probs(posts):
    processed = pd.DataFrame({
        "row_id": [r for r, _, _ in posts],
        "text": ["".join(t) for _, t, _ in posts],
        "label": [1 if TRIGGER in "".join(t) else 0 for _, t, _ in posts],
    })
    probs = pd.DataFrame({
        "row_id": processed["row_id"],
        "split": "test",
        "y_true": processed["label"],
        "y_prob": np.where(processed["label"] == 1, 0.9, 0.2),
    })
    return processed, probs


def _fake_model(threshold=0.5, offset=0.0):
    """P(EXTREMIST) is 0.9 when the trigger is present and 0.2 otherwise."""
    def predict(texts):
        return np.array([(0.9 if TRIGGER in t else 0.2) + offset for t in texts], dtype=float)

    return SimpleNamespace(technique="11_T", threshold=threshold, predict=predict, family="sklearn_pipeline")


def _frame(posts):
    processed, probs = _processed_and_probs(posts)
    return build_posts("11_T", posts, processed, probs)


class TestDeletionSets:
    def test_top_k_takes_positive_tokens_highest_first(self):
        values = np.array([0.1, -0.5, 0.3, 0.3, 0.0, 0.2])
        assert list(top_k_positive(values, 1)) == [2]
        assert list(top_k_positive(values, 3)) == [2, 3, 5]
        # Only four tokens are positive, so k=10 deletes four, never a negative one.
        assert list(top_k_positive(values, 10)) == [2, 3, 5, 0]

    def test_no_positive_token_deletes_nothing(self):
        assert len(top_k_positive(np.array([-0.1, 0.0]), 5)) == 0

    def test_random_is_seeded_and_matched(self):
        a = random_positions(10, 3, np.random.default_rng([30, 1]))
        b = random_positions(10, 3, np.random.default_rng([30, 1]))
        assert list(a) == list(b) and len(set(a)) == 3
        with pytest.raises(FaithfulnessError):
            random_positions(2, 3, np.random.default_rng(0))

    def test_delete_tokens_and_empty_post(self):
        tokens = ["we ", "should ", "burn ", "it"]
        assert delete_tokens(tokens, [2]) == "we should it"
        assert delete_tokens(tokens, [3]) == "we should burn"
        assert delete_tokens(tokens, [0, 1, 2, 3]) == MASK_TOKEN

    def test_plan_matches_random_count_to_top_k(self):
        frame = _frame(_posts(6))
        plan = deletion_plan(frame, k_values=(3,), seeds=(1, 2))
        # Trigger posts have 4 positive tokens, the others one.
        assert list(plan[3]["m"]) == [3, 1, 3, 1, 3, 1]
        assert len(plan[3]["random"]) == 2


class TestMetrics:
    def test_faithful_ranking_beats_random(self):
        frame = _frame(_posts())
        payload = analyze("11_T", _fake_model(), frame, 0.5, set(frame["row_id"]),
                          k_values=(1, 3), seeds=(1, 2, 3))
        row = next(r for r in payload["rows"] if r["population"] == "explained" and r["k"] == 1)
        # Deleting the trigger always clears the flagged posts; random rarely does.
        assert row["flip_share_topk"] == 1.0
        assert row["mean_drop_topk"] > row["mean_drop_random"]
        assert row["drop_minus_random"] > 0
        assert row["n_flagged"] == 20 and row["n_posts"] == 40

    def test_unfaithful_ranking_has_no_margin(self):
        posts = _posts()
        # Attribute everything to a word the model ignores.
        shuffled = [(r, t, np.where(np.arange(len(t)) == 0, 1.0, -0.1)) for r, t, _ in posts]
        frame = _frame(shuffled)
        payload = analyze("11_T", _fake_model(), frame, 0.5, set(frame["row_id"]),
                          k_values=(1,), seeds=(1, 2, 3))
        row = payload["rows"][0]
        assert row["mean_drop_topk"] == 0
        assert row["drop_minus_random"] <= 0

    def test_posts_without_positive_token_are_excluded(self):
        metrics = deletion_metrics([0.9, 0.2], [0.1, 0.2], np.array([[0.9, 0.2]]), [1, 0], 1, 0.5)
        assert metrics["n_posts"] == 1 and metrics["flip_share_topk"] == 1.0


class TestRefusals:
    def test_predictor_that_disagrees_with_the_artifact_is_refused(self):
        frame = _frame(_posts())
        with pytest.raises(FaithfulnessError, match="disagrees"):
            analyze("11_T", _fake_model(offset=0.05), frame, 0.5, set(frame["row_id"]), k_values=(1,))

    def test_threshold_mismatch_is_refused(self):
        frame = _frame(_posts())
        with pytest.raises(FaithfulnessError, match="threshold"):
            analyze("11_T", _fake_model(threshold=0.4), frame, 0.5, set(frame["row_id"]))

    def test_tokens_that_do_not_rebuild_the_text_are_refused(self):
        posts = _posts(4)
        processed, probs = _processed_and_probs(posts)
        processed.loc[0, "text"] = processed.loc[0, "text"] + " extra"
        with pytest.raises(FaithfulnessError, match="concatenate"):
            build_posts("11_T", posts, processed, probs)

    def test_explained_post_outside_the_artifact_is_refused(self):
        posts = _posts(4)
        processed, probs = _processed_and_probs(posts)
        with pytest.raises(FaithfulnessError, match="not a row"):
            build_posts("11_T", posts, processed, probs.iloc[1:])

    def test_per_post_file_that_does_not_rebuild_the_word_table_is_refused(self, tmp_path):
        posts = _posts(4)
        processed, probs = _processed_and_probs(posts)
        external, results = tmp_path / "external", tmp_path / "results_summary"
        local = fa.local_path("11_T", external)
        local.parent.mkdir(parents=True)
        pd.DataFrame([
            {"row_id": r, "position": i, "token": tok, "attribution": v}
            for r, toks, vals in posts for i, (tok, v) in enumerate(zip(toks, vals))
        ]).to_csv(local, index=False)
        table = aggregate_word_attributions([t for _, t, _ in posts], [v for _, _, v in posts])
        committed = results / "11_T" / "attributions" / local.name
        committed.parent.mkdir(parents=True)
        table.to_csv(committed, index=False)
        frame, diff = load_posts("11_T", processed, external, results, probs=probs)
        assert len(frame) == 4 and diff < 1e-9
        table.loc[0, "mean_attribution"] += 0.5
        table.to_csv(committed, index=False)
        with pytest.raises(FaithfulnessError, match="reproduce"):
            load_posts("11_T", processed, external, results, probs=probs)

    def test_missing_per_post_file_refuses_the_shared_population(self, tmp_path):
        results = tmp_path / "results_summary"
        (results / "01_A").mkdir(parents=True)
        with pytest.raises(FaithfulnessError, match="no per-post"):
            shared_row_ids(tmp_path / "external", results)

    def test_payload_with_a_row_id_is_never_written(self, tmp_path):
        frame = _frame(_posts())
        payload = analyze("11_T", _fake_model(), frame, 0.5, set(frame["row_id"]), k_values=(1,), seeds=(1,))
        check_text_free(payload)
        payload["note"] = "worst post ex_000004"
        with pytest.raises(FaithfulnessError, match="row id"):
            write_outputs(payload, tmp_path)
        assert not list(tmp_path.rglob("*.json"))
        with pytest.raises(FaithfulnessError):
            write_table([payload], tmp_path / "t.csv", tmp_path)
        assert not (tmp_path / "t.csv").exists()


class TestOutputs:
    def test_outputs_are_text_free_and_holm_is_per_population(self, tmp_path):
        frame = _frame(_posts())
        shared = set(frame["row_id"].iloc[:20])
        payloads = []
        for technique in ("01_A", "02_B"):
            model = _fake_model()
            model.technique = technique
            payloads.append(analyze(technique, model, frame, 0.5, shared, k_values=(1, 3), seeds=(1, 2)))
        results = tmp_path / "results_summary"
        results.mkdir()
        path = write_table(payloads, tmp_path / "rq2_faithfulness.csv", results)
        table = pd.read_csv(path)
        assert list(table.columns) == list(TABLE_COLUMNS)
        assert len(table) == 2 * 2 * 2
        assert set(table.loc[table["population"] == "shared", "n_posts_explained"]) == {20}
        for population, block in table.groupby("population"):
            p = block["wilcoxon_p"].to_numpy()
            assert (block["wilcoxon_p_holm"].to_numpy() >= p - 1e-12).all()
            assert block["wilcoxon_p_holm"].max() <= min(1.0, len(block) * p.max()) + 1e-12
        json_path = write_outputs(payloads[0], results)
        for file in (path, json_path):
            text = file.read_text()
            assert not ROW_ID_RE.search(text)
            assert "garden" not in text and TRIGGER not in text
        assert json.loads(json_path.read_text())["n_posts_explained"] == 40

    def test_rewriting_one_technique_keeps_the_others(self, tmp_path):
        frame = _frame(_posts())
        results = tmp_path / "results_summary"
        results.mkdir()
        path = tmp_path / "t.csv"
        for technique in ("01_A", "02_B", "01_A"):
            model = _fake_model()
            model.technique = technique
            write_table([analyze(technique, model, frame, 0.5, set(frame["row_id"]), k_values=(1,), seeds=(1,))],
                        path, results)
        assert sorted(pd.read_csv(path)["technique"].unique()) == ["01_A", "02_B"]
        assert len(pd.read_csv(path)) == 4
