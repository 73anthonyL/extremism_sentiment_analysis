"""Behavioural and adversarial tests for tools/audit.py on synthetic, text-free inputs."""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import audit  # noqa: E402
from conftest import FRAMING_WORDS, IDENTITY_WORDS, TOPICAL_WORDS  # noqa: E402

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
WORDS = FRAMING_WORDS + IDENTITY_WORDS + TOPICAL_WORDS + ["river", "garden", "window", "pencil"]
TECHNIQUES = ("01_A", "02_B")


def _lock_threshold(results_dir, technique, threshold=0.5, in_comparison=True):
    folder = results_dir / technique
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "best_config.json", "w") as handle:
        json.dump({"selected_threshold": threshold, "in_comparison": in_comparison}, handle)


def _write_check(results_dir, technique, run_id, passed, spearman=0.9):
    path = results_dir / technique / "attributions" / f"{run_id}__coefficient_check.json"
    path.write_text(json.dumps({
        "spearman_mean_attribution_vs_coefficient": spearman, "secondary_min_support": 2,
        "spearman_support_ge_min": spearman + 0.05, "n_shared_words_support_ge_min": 10,
        "top_k": 50, "top_k_sign_agreement": 1.0, "passed": passed,
    }))


def _contract_met(technique, make_probs, make_run, results_dir, seed=1):
    make_probs(technique, seed=seed)
    _lock_threshold(results_dir, technique)
    make_run(technique, "shap_test", WORDS, seed=seed, split="test")
    make_run(technique, "shap_validation", WORDS, seed=seed + 10, split="validation")


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


@pytest.fixture
def audited(tmp_path, make_probs, make_run, results_dir):
    """Two techniques meeting the contract, with rq tables built by the real tools."""
    import mcnemar_pairs
    import word_agreement

    for i, technique in enumerate(TECHNIQUES):
        _contract_met(technique, make_probs, make_run, results_dir, seed=i + 1)
    rq_dir = tmp_path / "rq"
    rq_dir.mkdir()

    pairs, overlap, shared, _, _ = mcnemar_pairs.build(list(TECHNIQUES), results_dir=results_dir)
    mcnemar_pairs.write_tables(pairs, overlap, shared, rq_dir / "rq1_mcnemar_pairs.csv",
                               rq_dir / "rq1_error_overlap.csv", rq_dir / "rq1_shared_errors.csv")
    row = pairs.iloc[0]
    for technique, accuracy in ((row["technique_a"], row["accuracy_a"]), (row["technique_b"], row["accuracy_b"])):
        with open(results_dir / technique / "metrics_test.json", "w") as handle:
            json.dump({"accuracy": float(accuracy)}, handle)

    agreement, top, support = word_agreement.build(results_dir=results_dir, min_shared_words=1)
    word_agreement.write_tables(agreement, top, support, rq_dir / "rq2_word_agreement_pairs.csv",
                                rq_dir / "rq2_shared_top_words.csv", rq_dir / "rq2_top_word_support.csv")
    pd.DataFrame({
        "technique": list(TECHNIQUES), "n_runs": [2, 2],
        "mean_pairwise_spearman_abs_attribution": [0.7, 0.4], "mean_pairwise_jaccard_top50": [0.4, 0.2],
    }).to_csv(rq_dir / "rq3_stability.csv", index=False)
    pd.DataFrame({
        "technique": list(TECHNIQUES), "split": "test", "threshold": 0.5, "n_negative": 280,
        "n_negative_with_identity_terms": 67, "fpr_with_identity_terms": [0.16, 0.13],
        "fpr_without_identity_terms": [0.08, 0.07], "fpr_ratio": [2.0, 1.9], "fisher_p_value": [0.07, 0.13],
    }).to_csv(rq_dir / "rq4_identity_fpr.csv", index=False)
    _write_fpr_sensitivity(rq_dir, pd.read_csv(rq_dir / "rq4_identity_fpr.csv"))
    return results_dir, rq_dir


def _write_fpr_sensitivity(rq_dir, rq4):
    """The sensitivity table identity_sensitivity.py fpr writes: full_lexicon equals RQ4, plus the primary variant."""
    full = rq4.assign(variant="full_lexicon", n_negative_excluded=0, n_positive_with_identity_terms=83,
                      n_positive_excluded=0, recall_without_identity_terms=0.8)
    excluded = full.assign(variant="exclude_ambiguous_only", n_negative_excluded=14, n_negative=266,
                           n_negative_with_identity_terms=53, n_positive_with_identity_terms=59,
                           n_positive_excluded=24, fpr_ratio=2.1, fisher_p_value=0.08)
    pd.concat([full, excluded], ignore_index=True).to_csv(rq_dir / "rq4_identity_fpr_sensitivity.csv", index=False)


def _write_restricted_cf(rq_dir, threshold=0.5, n_posts=None, technique="01_A"):
    """rq4_identity_counterfactual_sensitivity.csv rows for one technique (53 / 59 posts by default)."""
    n_posts = n_posts or {"non_extremist": 53, "extremist": 59}
    rows = [{"technique": technique, "split": "test", "population": name, "y_true": y, "operation": op,
             "threshold": threshold, "n_posts": n_posts[name], "mean_delta": 0.04, "n_flagged_original": 9,
             "n_flip_pos_to_neg": 3, "n_flip_neg_to_pos": 0, "wilcoxon_p_holm": 1e-4, "mcnemar_p_holm": 1.0}
            for name, y in (("non_extremist", 0), ("extremist", 1))
            for op in ("remove", "remove_unambiguous")]
    pd.DataFrame(rows).to_csv(rq_dir / "rq4_identity_counterfactual_sensitivity.csv", index=False)


def _write_faithfulness(results_dir, rq_dir, techniques, threshold=0.5):
    """faithfulness_test.json per technique and the cross-technique table, by the tool's own writers."""
    from faithfulness import ROW_COLUMNS, write_outputs, write_table
    from repo_paths import comparison_technique_dirs

    comparison = [d.name for d in comparison_technique_dirs(results_dir)]
    payloads = []
    for i, technique in enumerate(techniques):
        rows = []
        for population in ("explained", "shared"):
            for k in (1, 3):
                row = {c: 0.0 for c in ROW_COLUMNS}
                row.update(technique=technique, model_family="toy", split="test", population=population, k=k,
                           threshold=threshold, n_posts_explained=200, n_posts=198, n_flagged=70,
                           mean_drop_topk=0.1 * k, mean_drop_random=0.01 * k, drop_minus_random=0.09 * k,
                           mean_drop_topk_flagged=0.2, mean_drop_random_flagged=0.05,
                           wilcoxon_p=1e-6 * (i + 1) * k, n_random_seeds=5)
                rows.append(row)
        payload = {"technique": technique, "split": "test", "threshold": threshold, "n_posts_explained": 200,
                   "n_posts_shared": 200, "random_seeds": [30, 31, 32, 33, 34],
                   "verification": {"max_abs_diff": 1e-8}, "shared_population_from": comparison, "rows": rows}
        write_outputs(payload, results_dir)
        payloads.append(payload)
    return write_table(payloads, rq_dir / "rq2_faithfulness.csv", results_dir)


class TestCheckContract:
    def test_met_contract_has_no_failed_item(self, tmp_path, make_probs, make_run, results_dir, capsys):
        _contract_met("01_A", make_probs, make_run, results_dir)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        assert not [i for i in items if i.failed]
        assert audit.print_contract("01_A", items)
        assert "contract met" in capsys.readouterr().out

    def test_no_attribution_runs_fails_and_names_rq2_to_rq4(self, tmp_path, make_probs, results_dir, capsys):
        make_probs("01_A")
        _lock_threshold(results_dir, "01_A")
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        failed = [i for i in items if i.failed]
        assert [i.name for i in failed] == ["whole-model attribution run (test)",
                                            "whole-model attribution run (validation)"]
        assert all(i.status == audit.MISSING for i in failed)
        assert not audit.print_contract("01_A", items)
        assert "cannot take part in RQ2-RQ4" in capsys.readouterr().out

    def test_member_run_alone_does_not_satisfy_the_contract(self, tmp_path, make_probs, make_run, results_dir):
        make_probs("01_A")
        _lock_threshold(results_dir, "01_A")
        make_run("01_A", "shap_test_member", WORDS, member="M1", split="test")
        make_run("01_A", "shap_validation", WORDS, split="validation")
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        assert [i.name for i in items if i.failed] == ["whole-model attribution run (test)"]

    def test_truncated_probability_artifact_is_invalid(self, tmp_path, probs_dir, make_probs, make_run, results_dir):
        _contract_met("01_A", make_probs, make_run, results_dir)
        path = probs_dir / "01_A__test.csv"
        pd.read_csv(path).iloc[:-5].to_csv(path, index=False)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        item = next(i for i in items if i.name == "probability artifact (test)")
        assert item.status == audit.INVALID and item.failed
        assert "frozen protocol requires" in item.detail

    def test_text_column_in_probability_artifact_is_invalid(self, tmp_path, probs_dir, make_probs, make_run,
                                                            results_dir):
        _contract_met("01_A", make_probs, make_run, results_dir)
        path = probs_dir / "01_A__validation.csv"
        frame = pd.read_csv(path)
        frame["text"] = "placeholder"
        frame.to_csv(path, index=False)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        item = next(i for i in items if i.name == "probability artifact (validation)")
        assert item.status == audit.INVALID and "text-bearing" in item.detail

    def test_threshold_outside_unit_interval_is_invalid(self, tmp_path, make_probs, make_run, results_dir):
        _contract_met("01_A", make_probs, make_run, results_dir)
        _lock_threshold(results_dir, "01_A", threshold=1.5)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        assert next(i for i in items if i.name == "locked validation threshold").status == audit.INVALID

    def test_orphan_attribution_csv_is_invalid(self, tmp_path, make_probs, make_run, results_dir):
        _contract_met("01_A", make_probs, make_run, results_dir)
        (results_dir / "01_A" / "attributions" / "shap_test.json").unlink()
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        item = next(i for i in items if i.name == "whole-model attribution run (test)")
        assert item.status == audit.INVALID and "no sidecar" in item.detail

    def test_absent_optional_items_do_not_fail_but_broken_coefficients_do(self, tmp_path, make_probs, make_run,
                                                                         results_dir):
        _contract_met("01_A", make_probs, make_run, results_dir)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        optional = [i for i in items if not i.required]
        assert {i.status for i in optional} == {audit.MISSING} and not any(i.failed for i in optional)
        pd.DataFrame({"word": ["a"], "coefficient": [0.1], "row_id": ["x"]}).to_csv(
            results_dir / "01_A" / "attributions" / "coefficients.csv", index=False)
        items = audit.check_contract("01_A", results_dir=results_dir, external_dir=tmp_path / "external")
        item = next(i for i in items if i.name.startswith("coefficients.csv"))
        assert item.status == audit.INVALID and item.failed


class TestReport:
    def test_report_is_written_and_reads_the_tables(self, audited, tmp_path):
        results_dir, rq_dir = audited
        out = tmp_path / "cards"
        written = audit.write_reports(TECHNIQUES, results_dir=results_dir, rq_dir=rq_dir, out_dir=out)
        assert len(written) == 2
        payload = json.loads((out / "01_A.json").read_text())
        assert payload["rq1"]["pairs"][0]["other_technique"] == "02_B"
        assert payload["rq3"]["mean_pairwise_jaccard_top50"] == 0.4
        assert payload["rq4"]["n_negative_without_identity_terms"] == 213
        assert payload["wp7_counterfactual"]["status"] == "not run"
        assert payload["wp7_counterfactual"]["unambiguous_population"]["status"] == "not run"
        assert payload["faithfulness"]["status"] == "not run"
        assert payload["rq4_ambiguous_terms_sensitivity"]["n_negative_with_identity_terms"] == 53
        assert len(payload["rq2"]["top_words_by_positive_mean_attribution"]) == 10
        markdown = (out / "01_A.md").read_text()
        assert "## WP7" in markdown and "Not run" in markdown
        assert "## RQ2 support" in markdown and "--with-perturbation" in markdown
        assert "Ambiguous identity words" in markdown and "ratio 2.10" in markdown

    def test_sensitivity_table_disagreeing_with_rq4_is_stale(self, audited):
        """identity_sensitivity's full_lexicon row recomputes RQ4; a mismatch means one table is stale."""
        results_dir, rq_dir = audited
        path = rq_dir / "rq4_identity_fpr_sensitivity.csv"
        frame = pd.read_csv(path)
        frame.loc[(frame["technique"] == "01_A") & (frame["variant"] == "full_lexicon"),
                  "fpr_with_identity_terms"] = 0.2
        frame.to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match="full_lexicon row for 01_A differs.*stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_faithfulness_section_reads_the_json_and_the_table(self, audited):
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        payload = audit.build_report("01_A", results_dir, rq_dir)
        faith = payload["faithfulness"]
        assert faith["status"] == "run" and faith["population"] == "shared"
        assert [r["k"] for r in faith["rows"]] == [1, 3]
        assert faith["holm_family_size"] == 4  # two comparison techniques x two k on `shared`
        assert faith["rows"][1]["drop_minus_random"] == pytest.approx(0.27)
        markdown = audit.render_markdown(payload)
        assert "Holm runs over the 4 `shared` rows of the comparison techniques" in markdown
        assert "does not show the model is right to need them" in markdown
        assert "better" not in markdown.lower() and "outperform" not in markdown.lower()

    def test_faithfulness_json_without_table_rows_is_stale(self, audited):
        results_dir, rq_dir = audited
        path = _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        frame = pd.read_csv(path)
        frame[frame["technique"] != "02_B"].to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match="rq2_faithfulness.csv has no row for 02_B; the table is stale"):
            audit.build_report("02_B", results_dir, rq_dir)
        with pytest.raises(audit.AuditError, match="stale"):  # 01_A's card is refused too: its family lacks 02_B
            audit.write_reports(TECHNIQUES, results_dir=results_dir, rq_dir=rq_dir, out_dir=rq_dir / "cards")
        assert not (rq_dir / "cards").exists()

    def test_faithfulness_table_disagreeing_with_its_json_is_stale(self, audited):
        results_dir, rq_dir = audited
        path = _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        frame = pd.read_csv(path)
        frame.loc[(frame["technique"] == "01_A") & (frame["k"] == 3), "mean_drop_topk"] = 0.9
        frame.to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match=r"does not match .*faithfulness_test.json.*k=3 \(mean_drop_topk\); the table is stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_faithfulness_rows_without_their_json_are_stale(self, audited):
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        (results_dir / "01_A" / "faithfulness_test.json").unlink()
        with pytest.raises(audit.AuditError, match="faithfulness_test.json is missing; the table is stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_faithfulness_at_another_threshold_is_stale(self, audited):
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        _lock_threshold(results_dir, "01_A", threshold=0.6)  # the threshold was relocked after the run
        with pytest.raises(audit.AuditError, match="locked threshold is 0.6; it is stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_restricted_counterfactual_rows_join_the_wp7_table(self, audited):
        results_dir, rq_dir = audited
        _write_restricted_cf(rq_dir)
        payload = audit.build_report("01_A", results_dir, rq_dir)
        restricted = payload["wp7_counterfactual"]["unambiguous_population"]
        assert restricted["status"] == "run" and len(restricted["rows"]) == 2  # remove only
        markdown = audit.render_markdown(payload)
        assert "full-population counterfactual was not run" in markdown
        assert "| non_extremist, unambiguous match | 53 |" in markdown
        assert "not the WP7 family" in markdown
        assert "its effect size is that of the restricted posts" in markdown

    def test_restricted_counterfactual_at_another_threshold_is_stale(self, audited):
        results_dir, rq_dir = audited
        _write_restricted_cf(rq_dir, threshold=0.45)
        with pytest.raises(audit.AuditError, match=r"threshold \[0.45\] but the locked threshold is 0.5"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_restricted_counterfactual_population_must_match_the_sensitivity_counts(self, audited):
        """53 / 59 posts keep an unambiguous match in the FPR table; the counterfactual must cover those."""
        results_dir, rq_dir = audited
        _write_restricted_cf(rq_dir, n_posts={"non_extremist": 50, "extremist": 59})
        with pytest.raises(audit.AuditError, match="cover .* posts but .* one of them is stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_wp7_rows_at_another_threshold_are_stale(self, audited):
        results_dir, rq_dir = audited
        pd.DataFrame([{"technique": "01_A", "split": "test", "population": "non_extremist", "y_true": 0,
                       "operation": "remove", "threshold": 0.4, "n_posts": 67, "mean_delta": 0.03,
                       "n_flagged_original": 9, "n_flip_pos_to_neg": 3, "n_flip_neg_to_pos": 0,
                       "wilcoxon_p_holm": 1e-4, "mcnemar_p_holm": 1.0}]).to_csv(
            rq_dir / "rq4_identity_counterfactual.csv", index=False)
        with pytest.raises(audit.AuditError, match="rq4_identity_counterfactual.csv scored 01_A at threshold"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_excluded_row_inconsistent_with_full_lexicon_is_stale(self, audited):
        results_dir, rq_dir = audited
        path = rq_dir / "rq4_identity_fpr_sensitivity.csv"
        frame = pd.read_csv(path)
        frame.loc[(frame["technique"] == "01_A") & (frame["variant"] == "exclude_ambiguous_only"),
                  "fpr_without_identity_terms"] = 0.05  # excluded posts all have an identity match
        frame.to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match="not its full_lexicon row minus the excluded posts"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_nan_on_both_sides_is_equal_and_on_one_side_is_not(self):
        nan = float("nan")
        assert audit._same_number(nan, None) and audit._same_number(nan, nan)
        assert not audit._same_number(nan, 0.0) and not audit._same_number(0.0, None)

    def test_faithfulness_from_another_comparison_is_stale(self, audited):
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        path = results_dir / "01_A" / "faithfulness_test.json"
        payload = json.loads(path.read_text())
        payload["shared_population_from"] = ["01_A"]
        path.write_text(json.dumps(payload))
        with pytest.raises(audit.AuditError, match="took its shared population from"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_faithfulness_holm_not_over_the_current_family_is_stale(self, audited):
        results_dir, rq_dir = audited
        path = _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        frame = pd.read_csv(path)
        own = (frame["technique"] == "01_A") & (frame["population"] == "shared")
        frame.loc[own, "wilcoxon_p_holm"] = frame.loc[own, "wilcoxon_p"]  # left unadjusted
        frame.to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match=r"wilcoxon_p_holm for 01_A \(shared, k=\[1, 3\]\) is not Holm"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_faithfulness_for_part_of_the_comparison_is_stale(self, audited):
        """01_A's card is refused while 02_B has no faithfulness rows: the Holm family is incomplete."""
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, ["01_A"])
        with pytest.raises(audit.AuditError, match=r"none for \['02_B'\]; the Holm family is incomplete"):
            audit.build_report("01_A", results_dir, rq_dir)
        assert audit.build_report("02_B", results_dir, rq_dir)["faithfulness"]["status"] == "not run"

    @pytest.mark.parametrize("column", ["mean_attributed_random", "drop_minus_random_flagged"])
    def test_faithfulness_any_numeric_field_differing_is_stale(self, audited, column):
        results_dir, rq_dir = audited
        path = _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        frame = pd.read_csv(path)
        frame[column] = frame[column].astype(float)
        frame.loc[(frame["technique"] == "02_B") & (frame["k"] == 1), column] = 0.777
        frame.to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match=f"k=1 \\({column}\\); the table is stale"):
            audit.build_report("02_B", results_dir, rq_dir)

    def test_faithfulness_row_sets_must_be_equal(self, audited):
        results_dir, rq_dir = audited
        path = _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        frame = pd.read_csv(path)
        extra = frame[(frame["technique"] == "01_A") & (frame["k"] == 3)].assign(k=5)
        pd.concat([frame, extra]).to_csv(path, index=False)
        with pytest.raises(audit.AuditError, match="are not the \\(population, k\\) rows"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_card_names_n_of_the_shared_set(self, audited):
        results_dir, rq_dir = audited
        _write_faithfulness(results_dir, rq_dir, TECHNIQUES)
        payload = audit.build_report("01_A", results_dir, rq_dir)
        assert payload["faithfulness"]["n_posts_shared_set"] == 200
        assert "on 200 of the 200 test posts that every comparison technique explained" in \
            audit.render_markdown(payload)

    def test_rq1_gap_is_oriented_to_the_technique(self, audited):
        results_dir, rq_dir = audited
        a = audit.build_report("01_A", results_dir, rq_dir)["rq1"]["pairs"][0]
        b = audit.build_report("02_B", results_dir, rq_dir)["rq1"]["pairs"][0]
        assert a["row_gap"] == -b["row_gap"]

    def test_wording_never_ranks_an_insignificant_gap(self, audited):
        results_dir, rq_dir = audited
        payload = audit.build_report("01_A", results_dir, rq_dir)
        pair = dict(payload["rq1"]["pairs"][0], significant_holm=False, row_gap=5)
        phrase = audit._gap_phrase("01_A", pair)
        assert "not distinguishable at n = 450" in phrase
        markdown = audit.render_markdown(payload).lower()
        assert "better" not in markdown and "outperform" not in markdown

    @pytest.mark.parametrize("table", ["rq1_mcnemar_pairs.csv", "rq2_word_agreement_pairs.csv",
                                       "rq3_stability.csv", "rq4_identity_fpr.csv",
                                       "rq4_identity_fpr_sensitivity.csv"])
    def test_refuses_a_technique_missing_from_a_table(self, audited, tmp_path, table):
        results_dir, rq_dir = audited
        path = rq_dir / table
        frame = pd.read_csv(path)
        keep = pd.Series(True, index=frame.index)
        for column in ("technique", "technique_a", "technique_b"):
            if column in frame.columns:
                keep &= frame[column] != "02_B"
        frame[keep].to_csv(path, index=False)
        out = tmp_path / "cards"
        with pytest.raises(audit.AuditError, match="the table is stale"):
            audit.write_reports(TECHNIQUES, results_dir=results_dir, rq_dir=rq_dir, out_dir=out)
        assert not out.exists() or not list(out.iterdir())  # nothing written for anyone

    def test_report_cli_refuses_a_stale_table_and_writes_nothing(self, audited, tmp_path, monkeypatch, capsys):
        results_dir, rq_dir = audited
        path = rq_dir / "rq3_stability.csv"
        frame = pd.read_csv(path)
        frame[frame["technique"] != "01_A"].to_csv(path, index=False)
        out = tmp_path / "cards"
        monkeypatch.setattr(audit, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(audit, "RQ_DIR", rq_dir)
        monkeypatch.setattr(audit, "REPORT_CARDS_DIR", out)
        assert audit.main(["report", "--technique", "01_A"]) == audit.EXIT_FAILED
        assert "no report card was written" in capsys.readouterr().err
        assert not out.exists() or not list(out.iterdir())

    def test_report_cli_turns_a_malformed_input_into_exit_1(self, audited, tmp_path, monkeypatch, capsys):
        results_dir, rq_dir = audited
        (results_dir / "01_A" / "metrics_test.json").write_text(json.dumps({"no_accuracy": 1}))
        monkeypatch.setattr(audit, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(audit, "RQ_DIR", rq_dir)
        monkeypatch.setattr(audit, "REPORT_CARDS_DIR", tmp_path / "cards")
        assert audit.main(["report", "--technique", "01_A"]) == audit.EXIT_FAILED
        assert "unexpected KeyError" in capsys.readouterr().err

    def test_card_carries_the_shap_check_and_conditions_rq2_on_it(self, audited, monkeypatch):
        results_dir, rq_dir = audited
        monkeypatch.setattr(audit, "ANCHOR_TECHNIQUE", "01_A")
        _write_check(results_dir, "01_A", "shap_test", passed=True)
        _write_check(results_dir, "01_A", "shap_validation", passed=True)
        assert audit.build_report("02_B", results_dir, rq_dir)["shap_check"]["status"] == "no coefficients"

        pd.DataFrame({"word": WORDS, "coefficient": 0.1}).to_csv(
            results_dir / "02_B" / "attributions" / "coefficients.csv", index=False)
        with pytest.raises(audit.AuditError, match="the check is stale"):
            audit.build_report("02_B", results_dir, rq_dir)  # coefficients but no check file

        _write_check(results_dir, "02_B", "shap_test", passed=False, spearman=0.79)
        _write_check(results_dir, "02_B", "shap_validation", passed=True)
        payload = audit.build_report("02_B", results_dir, rq_dir)
        assert payload["shap_check"]["status"] == "FAIL"
        markdown = audit.render_markdown(payload)
        assert "## SHAP pipeline check" in markdown and "**FAIL**" in markdown
        assert "no cross-model claim about this technique's reliance should rest on them" in markdown

    def test_empty_slur_lexicon_is_stated_on_the_card(self, audited, monkeypatch):
        import lexicons

        results_dir, rq_dir = audited
        real = lexicons.load_lexicons()
        monkeypatch.setattr(lexicons, "load_lexicons", lambda *a, **k: dict(real, slur=[]))
        payload = audit.build_report("01_A", results_dir, rq_dir)
        assert payload["rq2"]["empty_lexicon_categories"] == ["slur"]
        assert "The slur lexicon is empty" in audit.render_markdown(payload)

    def test_top_words_are_positive_only(self, audited):
        results_dir, rq_dir = audited
        words = audit.build_report("01_A", results_dir, rq_dir)["rq2"]["top_words_by_positive_mean_attribution"]
        assert words and all(w["mean_attribution"] > 0 for w in words)

    def test_refuses_a_pair_table_missing_a_newer_technique(self, audited, make_probs, make_run):
        results_dir, rq_dir = audited
        _contract_met("03_C", make_probs, make_run, results_dir, seed=7)  # joined after the tables were built
        with pytest.raises(audit.AuditError, match=r"paired with \['03_C'\]; the table is stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_refuses_when_mcnemar_accuracy_disagrees_with_metrics(self, audited):
        results_dir, rq_dir = audited
        with open(results_dir / "01_A" / "metrics_test.json", "w") as handle:
            json.dump({"accuracy": 0.123}, handle)
        with pytest.raises(audit.AuditError, match="stale"):
            audit.build_report("01_A", results_dir, rq_dir)

    def test_output_has_no_row_id_key_and_no_row_ids(self, audited, tmp_path):
        results_dir, rq_dir = audited
        out = tmp_path / "cards"
        audit.write_reports(TECHNIQUES, results_dir=results_dir, rq_dir=rq_dir, out_dir=out)
        for path in out.iterdir():
            content = path.read_text().replace(str(tmp_path), "")
            assert not ROW_ID_RE.search(content), path.name
            assert "row_id" not in content
        payload = json.loads((out / "01_A.json").read_text())
        assert not set(_keys(payload)) & set(audit.FORBIDDEN_REPORT_KEYS)

    def test_a_row_keyed_payload_is_refused(self, audited, tmp_path):
        results_dir, rq_dir = audited
        payload = audit.build_report("01_A", results_dir, rq_dir)
        payload["rq4"]["row_id"] = ["x"]
        with pytest.raises(audit.AuditError, match="row_id"):
            audit.write_payload(payload, tmp_path / "cards")
        payload = audit.build_report("01_A", results_dir, rq_dir)
        payload["rq1"]["pairs"][0]["other_technique"] = "ex_000123"
        with pytest.raises(audit.AuditError, match="row id"):
            audit.write_payload(payload, tmp_path / "cards")

    def test_committed_card_contains_no_dataset_text(self, tmp_path):
        """The real 01 card, rendered from the committed tables, matches no dataset post."""
        from scan_text_leakage import find_leaked_texts, load_candidates, normalize_text

        if not (audit.RQ_DIR / "rq1_mcnemar_pairs.csv").exists():
            pytest.skip("committed rq tables not present")
        md_path, json_path = audit.write_reports([audit.ANCHOR_TECHNIQUE], out_dir=tmp_path)[0]
        candidates, _, _ = load_candidates()
        for path in (md_path, json_path):
            content = path.read_text()
            count, _ = find_leaked_texts(normalize_text(content), candidates)
            assert count == 0, path.name
            assert not ROW_ID_RE.search(content)


class TestRun:
    def _record_stages(self, monkeypatch, fail_at=None):
        called = []
        for name in audit.STAGE_ORDER:
            def stage(ctx, name=name):
                called.append(name)
                if name == fail_at:
                    raise audit.StageFailed(f"{name} broke on purpose")
                return "ok"
            monkeypatch.setattr(audit, f"stage_{name}", stage)
        return called

    def test_stage_order_is_the_documented_order(self):
        order = audit.STAGE_ORDER
        assert order[0] == "validate_results_folder" and order[-1] == "report"
        assert order.index("validate_shap") < order.index("categorize_attributions") \
            < order.index("compare_reliance") < order.index("identity_fpr") < order.index("render_tables")

    def test_full_chain_runs_every_stage(self, tmp_path, make_probs, make_run, results_dir, monkeypatch):
        _contract_met("01_A", make_probs, make_run, results_dir)
        called = self._record_stages(monkeypatch)
        ok, failed = audit.run_chain(audit.Context("01_A", results_dir=results_dir), tmp_path / "external")
        assert ok and failed is None and called == list(audit.STAGE_ORDER)

    def test_stops_at_and_names_the_first_failing_stage(self, tmp_path, make_probs, make_run, results_dir,
                                                        monkeypatch, capsys):
        _contract_met("01_A", make_probs, make_run, results_dir)
        called = self._record_stages(monkeypatch, fail_at="compare_reliance")
        ok, failed = audit.run_chain(audit.Context("01_A", results_dir=results_dir), tmp_path / "external")
        assert not ok and failed == "compare_reliance"
        assert called == list(audit.STAGE_ORDER[: audit.STAGE_ORDER.index("compare_reliance") + 1])
        assert "STOPPED at stage compare_reliance" in capsys.readouterr().out

    def test_an_unexpected_exception_still_names_the_stage(self, tmp_path, make_probs, make_run, results_dir,
                                                           monkeypatch, capsys):
        _contract_met("01_A", make_probs, make_run, results_dir)
        called = self._record_stages(monkeypatch)

        def broken(ctx):
            called.append("identity_fpr")
            raise KeyError("accuracy")

        monkeypatch.setattr(audit, "stage_identity_fpr", broken)
        ok, failed = audit.run_chain(audit.Context("01_A", results_dir=results_dir), tmp_path / "external")
        assert not ok and failed == "identity_fpr"
        assert called[-1] == "identity_fpr" and "mcnemar_pairs" not in called
        out = capsys.readouterr().out
        assert "STOPPED at stage identity_fpr: unexpected KeyError" in out

    def test_contract_failure_stops_before_any_stage(self, tmp_path, make_probs, results_dir, monkeypatch):
        make_probs("01_A")
        _lock_threshold(results_dir, "01_A")
        called = self._record_stages(monkeypatch)
        ok, failed = audit.run_chain(audit.Context("01_A", results_dir=results_dir), tmp_path / "external")
        assert not ok and failed == "check-contract" and called == []

    def test_counterfactual_needs_the_external_folder(self, tmp_path, make_probs, make_run, results_dir, monkeypatch):
        _contract_met("01_A", make_probs, make_run, results_dir)
        called = self._record_stages(monkeypatch)
        ctx = audit.Context("01_A", with_perturbation=True, results_dir=results_dir)
        ok, failed = audit.run_chain(ctx, tmp_path / "external")
        assert not ok and failed == "check-contract" and called == []

    def test_a_failing_tool_exit_is_a_stage_failure(self):
        with pytest.raises(audit.StageFailed, match="exited 1"):
            audit._tool("validate_results_folder.py", "--technique", "99_NO-SUCH-TECHNIQUE")

    def test_new_stages_sit_after_their_inputs(self):
        order = audit.STAGE_ORDER
        assert order.index("identity_fpr") < order.index("identity_sensitivity_fpr")
        assert order.index("categorize_attributions") < order.index("same_post_stability")
        for stage in audit.PERTURBATION_STAGES:
            assert order.index("same_post_stability") < order.index(stage) < order.index("rq2_rq4_relation")
        # faithfulness writes into the technique folder, so the manifest refresh comes after it.
        assert order.index("rq2_rq4_relation") < order.index("refresh_manifest") < order.index("render_tables")

    def test_cli_flag_and_its_alias_set_the_same_option(self, monkeypatch):
        seen = []
        monkeypatch.setattr(audit, "run_chain", lambda ctx: (seen.append(ctx.with_perturbation), (True, None))[1])
        for flag in ("--with-perturbation", "--with-counterfactual"):
            assert audit.main(["run", "--technique", "01_A", flag]) == audit.EXIT_OK
        assert audit.main(["run", "--technique", "01_A"]) == audit.EXIT_OK
        assert seen == [True, True, False]


def _no_tool(monkeypatch):
    """Make any tool call fail the test: a skipped stage must not reach its tool."""
    def refuse(script, *args):
        raise AssertionError(f"{script} was called by a stage that should have skipped")
    monkeypatch.setattr(audit, "_tool", refuse)


class TestSkippedStages:
    """A stage whose inputs are absent says SKIP and why; it never reports OK."""

    @pytest.fixture
    def two(self, tmp_path, make_probs, make_run, results_dir):
        for i, technique in enumerate(TECHNIQUES):
            make_probs(technique, seed=i + 1)
            _lock_threshold(results_dir, technique)
            make_run(technique, "shap-partition_test_seed30", WORDS, seed=i + 1, split="test")
            make_run(technique, "shap-partition_validation_seed30", WORDS, seed=i + 11, split="validation")
        external = tmp_path / "external"
        return results_dir, external

    @staticmethod
    def _local(external, technique, split="test"):
        path = external / technique / "attributions_local" / f"shap-partition_{split}_seed30.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("row_id,position,token,attribution\n")
        return path

    @pytest.mark.parametrize("stage", audit.PERTURBATION_STAGES)
    def test_perturbation_stages_skip_without_the_flag(self, two, monkeypatch, stage):
        results_dir, external = two
        _no_tool(monkeypatch)
        result = getattr(audit, f"stage_{stage}")(audit.Context("01_A", results_dir=results_dir,
                                                                external_dir=external))
        assert isinstance(result, audit.Skip) and result.reason == audit.NOT_REQUESTED

    def test_faithfulness_without_per_post_files_is_a_skip_naming_them(self, two, monkeypatch):
        results_dir, external = two
        _no_tool(monkeypatch)
        self._local(external, "01_A")  # its own file is there; 02_B's, needed for the shared posts, is not
        ctx = audit.Context("01_A", with_perturbation=True, results_dir=results_dir, external_dir=external)
        result = audit.stage_faithfulness(ctx)
        assert isinstance(result, audit.Skip)
        assert "1 per-post attribution file(s) absent" in result.reason and "02_B" in result.reason
        assert "rq2_faithfulness.csv is left as is" in result.reason

    def test_faithfulness_runs_once_every_file_is_there(self, two, monkeypatch):
        results_dir, external = two
        for technique in TECHNIQUES:
            self._local(external, technique)
        calls = []
        monkeypatch.setattr(audit, "_tool", lambda script, *args: calls.append((script, args)) or "ok")
        ctx = audit.Context("01_A", with_perturbation=True, device="mps", sequential_checkpoints=True,
                            results_dir=results_dir, external_dir=external)
        assert audit.stage_faithfulness(ctx) == "ok"
        assert calls == [("faithfulness.py", ("--technique", "01_A", "--device", "mps",
                                              "--sequential-checkpoints"))]

    def test_faithfulness_skips_a_run_it_cannot_read(self, two, make_run, make_probs, monkeypatch):
        results_dir, external = two
        _no_tool(monkeypatch)
        make_probs("03_C", seed=5)
        _lock_threshold(results_dir, "03_C")
        make_run("03_C", "shap_other_name", WORDS, split="test")
        ctx = audit.Context("03_C", with_perturbation=True, results_dir=results_dir, external_dir=external)
        result = audit.stage_faithfulness(ctx)
        assert isinstance(result, audit.Skip) and "shap-partition_test_seed30" in result.reason

    def test_missing_per_post_files_are_recorded_as_skip_not_ok(self, two, monkeypatch, capsys):
        """The whole chain: faithfulness requested, files absent, so SKIP is printed and listed."""
        results_dir, external = two
        for name in audit.STAGE_ORDER:
            if name != "faithfulness":
                monkeypatch.setattr(audit, f"stage_{name}", lambda ctx: "ok")
        _no_tool(monkeypatch)
        monkeypatch.setattr(audit, "_check_external",
                            lambda t, e: audit.ContractItem("restorable predictor under external/", audit.OK, "x",
                                                            required=False))
        ctx = audit.Context("01_A", with_perturbation=True, results_dir=results_dir)
        ok, failed = audit.run_chain(ctx, external)
        out = capsys.readouterr().out
        assert ok and failed is None
        assert "SKIP  faithfulness: 2 per-post attribution file(s) absent" in out
        assert "OK    faithfulness" not in out
        assert [name for name, _ in ctx.skipped] == ["faithfulness"]
        assert "1 stage(s) skipped: faithfulness" in out

    def test_same_post_stability_skips_without_every_comparison_file(self, two, monkeypatch):
        results_dir, external = two
        _no_tool(monkeypatch)
        for technique in TECHNIQUES:
            self._local(external, technique, "test")
        self._local(external, "01_A", "validation")
        result = audit.stage_same_post_stability(audit.Context("01_A", results_dir=results_dir,
                                                               external_dir=external))
        assert isinstance(result, audit.Skip) and "02_B" in result.reason and "left as is" in result.reason

    def test_same_post_stability_and_the_relation_skip_an_outside_technique(self, two, make_probs, make_run,
                                                                           monkeypatch):
        results_dir, external = two
        _no_tool(monkeypatch)
        _contract_met("98_TOY", make_probs, make_run, results_dir, seed=9)
        _lock_threshold(results_dir, "98_TOY", in_comparison=False)
        for technique in (*TECHNIQUES, "98_TOY"):
            for split in ("test", "validation"):
                self._local(external, technique, split)
        ctx = audit.Context("98_TOY", results_dir=results_dir, external_dir=external)
        for stage in (audit.stage_same_post_stability, audit.stage_rq2_rq4_relation):
            result = stage(ctx)
            assert isinstance(result, audit.Skip) and "outside the comparison" in result.reason

    def test_relation_skips_while_an_input_lacks_a_comparison_technique(self, two, tmp_path, monkeypatch):
        results_dir, _ = two
        _no_tool(monkeypatch)
        rq_dir = tmp_path / "rq"
        rq_dir.mkdir()
        monkeypatch.setattr(audit, "RQ_DIR", rq_dir)
        pd.DataFrame({"technique": ["01_A"], "population": "non_extremist", "operation": "remove",
                      "threshold": 0.5}).to_csv(rq_dir / "rq4_identity_counterfactual.csv", index=False)
        pd.DataFrame({"technique": list(TECHNIQUES), "post_set": "common"}).to_csv(
            rq_dir / "rq2_same_post_category_shares.csv", index=False)
        result = audit.stage_rq2_rq4_relation(audit.Context("01_A", results_dir=results_dir))
        assert isinstance(result, audit.Skip)
        assert "rq4_identity_counterfactual.csv has no row for ['02_B']" in result.reason
        assert "stale" in result.reason

    def test_relation_skips_when_same_post_stability_skipped_this_run(self, two, monkeypatch):
        results_dir, _ = two
        _no_tool(monkeypatch)
        ctx = audit.Context("01_A", results_dir=results_dir, skipped=[("same_post_stability", "files absent")])
        result = audit.stage_rq2_rq4_relation(ctx)
        assert isinstance(result, audit.Skip) and "same_post_stability was skipped in this run" in result.reason

    def test_relation_skips_while_a_counterfactual_threshold_is_not_the_locked_one(self, two, tmp_path,
                                                                                 monkeypatch):
        results_dir, _ = two
        _no_tool(monkeypatch)
        rq_dir = tmp_path / "rq"
        rq_dir.mkdir()
        monkeypatch.setattr(audit, "RQ_DIR", rq_dir)
        pd.DataFrame({"technique": list(TECHNIQUES), "population": "non_extremist", "operation": "remove",
                      "threshold": [0.5, 0.42]}).to_csv(rq_dir / "rq4_identity_counterfactual.csv", index=False)
        pd.DataFrame({"technique": list(TECHNIQUES), "post_set": "common"}).to_csv(
            rq_dir / "rq2_same_post_category_shares.csv", index=False)
        result = audit.stage_rq2_rq4_relation(audit.Context("01_A", results_dir=results_dir))
        assert isinstance(result, audit.Skip) and "other than the locked one for ['02_B']" in result.reason

    def test_requested_perturbation_that_skips_exits_2(self, monkeypatch, capsys):
        def chain(ctx):
            ctx.skipped.append(("faithfulness", "2 per-post attribution file(s) absent"))
            return True, None

        monkeypatch.setattr(audit, "run_chain", chain)
        assert audit.main(["run", "--technique", "01_A", "--with-perturbation"]) == audit.EXIT_PERTURBATION_SKIPPED
        assert "faithfulness did not run" in capsys.readouterr().out
        # Without the flag the same skip is the expected "not requested" and the run exits 0.
        monkeypatch.setattr(audit, "run_chain",
                            lambda ctx: (ctx.skipped.append(("faithfulness", audit.NOT_REQUESTED)), (True, None))[1])
        assert audit.main(["run", "--technique", "01_A"]) == audit.EXIT_OK
        assert audit.main(["run", "--technique", "01_A", "--with-perturbation"]) == audit.EXIT_OK

    def test_without_the_flag_no_model_loading_tool_is_ever_called(self, two, tmp_path, monkeypatch, capsys):
        """Every real stage that calls a tool, with a recording _tool: no perturbation tool appears."""
        results_dir, external = two
        calls = []
        monkeypatch.setattr(audit, "_tool", lambda script, *args: calls.append((script, *args)) or "ok")
        monkeypatch.setattr(audit, "ANCHOR_TECHNIQUE", "01_A")
        monkeypatch.setattr(audit, "RQ_DIR", tmp_path / "rq")
        for name in ("validate_results_folder", "refresh_manifest", "report"):  # no _tool call; need real files
            monkeypatch.setattr(audit, f"stage_{name}", lambda ctx: "ok")
        for technique in TECHNIQUES:
            for split in ("test", "validation"):
                self._local(external, technique, split)  # so the stages that need files are not skipped for that
        ctx = audit.Context("01_A", with_perturbation=False, results_dir=results_dir)
        ok, _ = audit.run_chain(ctx, external)
        assert ok
        scripts = [c[0] for c in calls]
        assert "faithfulness.py" not in scripts and "identity_counterfactual.py" not in scripts
        assert not [c for c in calls if c[:2] == ("identity_sensitivity.py", "counterfactual")]
        assert "same_post_stability.py" in scripts and ("identity_sensitivity.py", "fpr", "--technique", "01_A") in calls
        assert [n for n, _ in ctx.skipped if n in audit.PERTURBATION_STAGES] == list(audit.PERTURBATION_STAGES)
        assert "perturbation stages skipped this run" in capsys.readouterr().out


class TestValidateShapStage:
    def _setup(self, make_probs, make_run, results_dir, monkeypatch, passed):
        import validate_shap

        _contract_met("01_A", make_probs, make_run, results_dir)
        pd.DataFrame({"word": WORDS, "coefficient": 0.1}).to_csv(
            results_dir / "01_A" / "attributions" / "coefficients.csv", index=False)
        monkeypatch.setattr(audit, "ANCHOR_TECHNIQUE", "01_A")
        stats = {"spearman_mean_attribution_vs_coefficient": 0.5, "top_k": 50,
                 "top_k_sign_agreement": 0.7, "passed": passed}
        monkeypatch.setattr(validate_shap, "validate", lambda *a, **k: (stats, None))
        return audit.Context("01_A", results_dir=results_dir)

    def test_a_fail_on_the_anchor_stops_the_chain(self, make_probs, make_run, results_dir, monkeypatch):
        ctx = self._setup(make_probs, make_run, results_dir, monkeypatch, passed=False)
        with pytest.raises(audit.StageFailed, match="FAIL on the anchor"):
            audit.stage_validate_shap(ctx)

    def test_a_fail_on_another_technique_is_recorded_not_fatal(self, make_probs, make_run, results_dir,
                                                               monkeypatch):
        """An ordinary word-level linear model that misses the bar still gets a card, flagged."""
        import validate_shap

        _contract_met("01_A", make_probs, make_run, results_dir)
        _write_check(results_dir, "01_A", "shap_test", passed=True)
        _contract_met("98_TOY", make_probs, make_run, results_dir, seed=5)
        pd.DataFrame({"word": WORDS, "coefficient": 0.1}).to_csv(
            results_dir / "98_TOY" / "attributions" / "coefficients.csv", index=False)
        monkeypatch.setattr(audit, "ANCHOR_TECHNIQUE", "01_A")
        stats = {"spearman_mean_attribution_vs_coefficient": 0.795, "top_k": 50, "top_k_sign_agreement": 1.0,
                 "passed": False, "secondary_min_support": 2, "spearman_support_ge_min": 0.9}
        monkeypatch.setattr(validate_shap, "validate", lambda *a, **k: (stats, None))
        message = audit.stage_validate_shap(audit.Context("98_TOY", results_dir=results_dir))
        assert "FAIL" in message and "Recorded on the report card" in message
        assert "CALIBRATION_POINT_TECHNIQUES" not in message and "tools/audit.py" not in message

    def test_fail_is_a_calibration_point_for_a_recorded_technique(self, make_probs, make_run, results_dir,
                                                                  monkeypatch):
        ctx = self._setup(make_probs, make_run, results_dir, monkeypatch, passed=False)
        monkeypatch.setitem(audit.CALIBRATION_POINT_TECHNIQUES, "01_A", "test reason")
        assert "recorded as a calibration point" in audit.stage_validate_shap(ctx)

    def test_other_techniques_need_the_anchor_to_pass(self, make_probs, make_run, results_dir, monkeypatch):
        _contract_met("02_B", make_probs, make_run, results_dir)
        monkeypatch.setattr(audit, "ANCHOR_TECHNIQUE", "01_A")
        with pytest.raises(audit.StageFailed, match="anchor"):
            audit.stage_validate_shap(audit.Context("02_B", results_dir=results_dir))


class TestOutsideComparison:
    """A technique with in_comparison false must not move the paper's families or tables."""

    @pytest.fixture
    def three(self, tmp_path, make_probs, make_run, results_dir):
        for i, technique in enumerate(TECHNIQUES):
            _contract_met(technique, make_probs, make_run, results_dir, seed=i + 1)
        _contract_met("98_TOY", make_probs, make_run, results_dir, seed=9)
        _lock_threshold(results_dir, "98_TOY", in_comparison=False)
        return results_dir

    def test_comparison_dirs_leave_out_the_outside_technique(self, three):
        from repo_paths import comparison_technique_dirs, outside_comparison_technique_dirs

        assert [d.name for d in comparison_technique_dirs(three)] == list(TECHNIQUES)
        assert [d.name for d in outside_comparison_technique_dirs(three)] == ["98_TOY"]

    def test_mcnemar_family_is_fixed_and_outside_pairs_have_their_own(self, three):
        import mcnemar_pairs

        alone, _, _, _, _ = mcnemar_pairs.build(list(TECHNIQUES), results_dir=three)
        default, overlap, _, _, _ = mcnemar_pairs.build(results_dir=three)
        pd.testing.assert_frame_equal(alone, default)
        assert set(overlap["technique"]) == set(TECHNIQUES)
        outside = mcnemar_pairs.build_outside(["98_TOY"], list(TECHNIQUES), results_dir=three)
        assert len(outside) == 2 and set(outside["technique_a"]) == {"98_TOY"}
        expected = mcnemar_pairs.holm_adjust(outside["mcnemar_p"])
        assert list(outside["holm_p"]) == pytest.approx(list(expected))

    def test_word_agreement_comparison_tables_leave_it_out(self, three):
        import word_agreement

        pairs, shared, support = word_agreement.build(results_dir=three, min_shared_words=1)
        assert "98_TOY" not in set(pairs["technique_a"]) | set(pairs["technique_b"])
        assert "98_TOY" not in set(support["technique"])
        outside = word_agreement.build_outside(results_dir=three, min_shared_words=1)
        assert len(outside) == 2 and set(outside["technique_a"]) == {"98_TOY"}

    def test_counterfactual_holm_family_excludes_the_outside_technique(self, three, tmp_path):
        from identity_counterfactual import ROW_COLUMNS, write_table

        def payload(technique, p):
            rows = []
            for y_true, operation in ((0, "remove"), (1, "remove")):
                row = {c: 0 for c in ROW_COLUMNS}
                row.update(technique=technique, split="test", population=f"p{y_true}", y_true=y_true,
                           operation=operation, wilcoxon_p=p, mcnemar_exact_p=p)
                rows.append(row)
            return {"rows": rows}

        inside = [payload("01_A", 0.01), payload("02_B", 0.02)]
        before = pd.read_csv(write_table(inside, path=tmp_path / "a.csv", results_dir=three))
        after = pd.read_csv(write_table([*inside, payload("98_TOY", 0.001)], path=tmp_path / "b.csv",
                                        results_dir=three))
        kept = after[after["technique"] != "98_TOY"].reset_index(drop=True)
        assert list(kept["mcnemar_p_holm"]) == pytest.approx(list(before["mcnemar_p_holm"]))
        assert list(after[after["technique"] == "98_TOY"]["mcnemar_p_holm"]) == pytest.approx([0.002, 0.002])

    def test_card_for_the_outside_technique_reads_its_own_tables(self, three, tmp_path):
        import mcnemar_pairs
        import word_agreement

        rq_dir = tmp_path / "rq"
        rq_dir.mkdir()
        pairs, overlap, shared, _, _ = mcnemar_pairs.build(results_dir=three)
        mcnemar_pairs.write_tables(pairs, overlap, shared, rq_dir / "rq1_mcnemar_pairs.csv",
                                   rq_dir / "rq1_error_overlap.csv", rq_dir / "rq1_shared_errors.csv")
        outside = mcnemar_pairs.build_outside(["98_TOY"], list(TECHNIQUES), results_dir=three)
        mcnemar_pairs.write_outside(outside, rq_dir / "rq1_mcnemar_pairs_outside_comparison.csv")
        accuracy = {outside.iloc[0]["technique_a"]: outside.iloc[0]["accuracy_a"]}
        for _, row in pairs.iterrows():
            accuracy[row["technique_a"]], accuracy[row["technique_b"]] = row["accuracy_a"], row["accuracy_b"]
        for technique, value in accuracy.items():
            (three / technique / "metrics_test.json").write_text(json.dumps({"accuracy": float(value)}))
        agreement, top, support = word_agreement.build(results_dir=three, min_shared_words=1)
        word_agreement.write_tables(agreement, top, support, rq_dir / "rq2_word_agreement_pairs.csv",
                                    rq_dir / "rq2_shared_top_words.csv", rq_dir / "rq2_top_word_support.csv")
        word_agreement.write_outside(word_agreement.build_outside(results_dir=three, min_shared_words=1),
                                     rq_dir / "rq2_word_agreement_pairs_outside_comparison.csv")
        everyone = [*TECHNIQUES, "98_TOY"]
        pd.DataFrame({"technique": everyone, "n_runs": 2, "mean_pairwise_spearman_abs_attribution": 0.5,
                      "mean_pairwise_jaccard_top50": 0.3}).to_csv(rq_dir / "rq3_stability.csv", index=False)
        pd.DataFrame({
            "technique": everyone, "split": "test", "threshold": 0.5, "n_negative": 280,
            "n_negative_with_identity_terms": 67, "fpr_with_identity_terms": 0.1,
            "fpr_without_identity_terms": 0.05, "fpr_ratio": 2.0, "fisher_p_value": 0.1,
        }).to_csv(rq_dir / "rq4_identity_fpr.csv", index=False)
        _write_fpr_sensitivity(rq_dir, pd.read_csv(rq_dir / "rq4_identity_fpr.csv"))
        _write_faithfulness(three, rq_dir, everyone)

        toy = audit.build_report("98_TOY", three, rq_dir)
        assert not toy["rq1"]["in_comparison"] and toy["rq1"]["holm_family_size"] == 2
        assert toy["faithfulness"]["holm_family_size"] == 2  # its own two `shared` rows
        markdown = audit.render_markdown(toy)
        assert "outside the comparison" in markdown
        assert "own 2 `shared` rows, so it does not move the comparison's p values" in markdown
        anchor = audit.build_report("01_A", three, rq_dir)  # not stale although 98_TOY is absent
        assert anchor["rq1"]["holm_family_size"] == 1
        assert [p["other_technique"] for p in anchor["rq1"]["pairs"]] == ["02_B"]
        assert anchor["faithfulness"]["holm_family_size"] == 4

    def test_faithfulness_holm_ignores_the_outside_technique(self, three, tmp_path):
        rq_dir = tmp_path / "rq"
        rq_dir.mkdir()
        before = pd.read_csv(_write_faithfulness(three, rq_dir, TECHNIQUES))
        after = pd.read_csv(_write_faithfulness(three, rq_dir, [*TECHNIQUES, "98_TOY"]))
        kept = after[after["technique"] != "98_TOY"].reset_index(drop=True)
        pd.testing.assert_frame_equal(kept, before)

    def test_two_outside_techniques_each_get_a_family_of_their_own(self, three, tmp_path, make_probs, make_run):
        """A second outside technique must not join the first one's family, nor the comparison's."""
        _contract_met("99_TOY2", make_probs, make_run, three, seed=11)
        _lock_threshold(three, "99_TOY2", in_comparison=False)
        rq_dir = tmp_path / "rq"
        rq_dir.mkdir()
        _write_faithfulness(three, rq_dir, [*TECHNIQUES, "98_TOY", "99_TOY2"])
        for technique in ("98_TOY", "99_TOY2"):
            faith = audit._faithfulness(technique, three, rq_dir)
            assert faith["holm_family_size"] == 2 and not faith["in_comparison"]
        assert audit._faithfulness("01_A", three, rq_dir)["holm_family_size"] == 4
