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
    return results_dir, rq_dir


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
        assert len(payload["rq2"]["top_words_by_positive_mean_attribution"]) == 10
        markdown = (out / "01_A.md").read_text()
        assert "## WP7" in markdown and "Not run" in markdown

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
                                       "rq3_stability.csv", "rq4_identity_fpr.csv"])
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
        ctx = audit.Context("01_A", with_counterfactual=True, results_dir=results_dir)
        ok, failed = audit.run_chain(ctx, tmp_path / "external")
        assert not ok and failed == "check-contract" and called == []

    def test_a_failing_tool_exit_is_a_stage_failure(self):
        with pytest.raises(audit.StageFailed, match="exited 1"):
            audit._tool("validate_results_folder.py", "--technique", "99_NO-SUCH-TECHNIQUE")


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

        toy = audit.build_report("98_TOY", three, rq_dir)
        assert not toy["rq1"]["in_comparison"] and toy["rq1"]["holm_family_size"] == 2
        assert "outside the comparison" in audit.render_markdown(toy)
        anchor = audit.build_report("01_A", three, rq_dir)  # not stale although 98_TOY is absent
        assert anchor["rq1"]["holm_family_size"] == 1
        assert [p["other_technique"] for p in anchor["rq1"]["pairs"]] == ["02_B"]
