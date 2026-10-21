"""Tests for tools/curated_plots.py.

Synthetic text only. A hand-built explain output folder (two techniques, three
texts, attributions that add up exactly) is drawn end to end. Adversarial tests
prove the tool refuses: predictions written before the explainer endpoints
were recorded, a base value that is not a probability, token positions with a
gap, and attributions that no longer add up to the explained output (which
would make every waterfall misstate the model).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

pytest.importorskip("shap")
pytest.importorskip("matplotlib")

import curated_examples as ce  # noqa: E402
import curated_plots as cp  # noqa: E402

TECHNIQUES = ("01_T", "07_T")
TEXTS = (
    "alarm one filler two alarm three",
    "calm words only here",
    "filler alarm calm mixed text.",
)
LABELS = (ce.POSITIVE_CLASS_NAME, ce.NEGATIVE_CLASS_NAME, ce.POSITIVE_CLASS_NAME)


def write_outputs(out_dir, techniques=TECHNIQUES):
    """An explain output folder whose attributions add up to the explained outputs."""
    out_dir.mkdir(parents=True)
    texts_file = out_dir.parent / "curated.txt"
    texts_file.write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
    labels_file = out_dir.parent / "labels.txt"
    labels_file.write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    examples = ce.read_examples(texts_file, labels_file)
    examples.to_csv(out_dir / ce.EXAMPLES_FILENAME, index=False)
    rng = np.random.default_rng(3)
    for t_index, technique in enumerate(techniques):
        folder = out_dir / technique
        folder.mkdir()
        base = 0.2 + 0.1 * t_index
        prediction_rows, attribution_rows = [], []
        for example_id, text in zip(examples["example_id"], examples["text"]):
            tokens = [word + " " for word in text.split()]
            values = rng.normal(0, 0.08, size=len(tokens))
            explained = float(np.clip(base + values.sum(), 0.01, 0.99))
            values *= (explained - base) / values.sum()
            raw = explained + (0.02 if text.endswith(".") else 0.0)
            prediction_rows.append(
                {
                    "example_id": example_id,
                    "technique": technique,
                    "member": ce.WHOLE_MODEL,
                    "y_prob": raw,
                    "threshold": 0.5,
                    "y_pred": int(raw >= 0.5),
                    ce.BASE_VALUE_COLUMN: base,
                    ce.EXPLAINED_OUTPUT_COLUMN: explained,
                }
            )
            for position, (token, value) in enumerate(zip(tokens, values)):
                attribution_rows.append(
                    {
                        "example_id": example_id,
                        "technique": technique,
                        "member": ce.WHOLE_MODEL,
                        "position": position,
                        "token": token,
                        "attribution": float(value),
                        "category": "topical",
                    }
                )
        pd.DataFrame(prediction_rows, columns=list(ce.PREDICTION_COLUMNS)).to_csv(
            folder / ce.PREDICTIONS_FILENAME, index=False
        )
        pd.DataFrame(attribution_rows, columns=list(ce.ATTRIBUTION_COLUMNS)).to_csv(
            folder / ce.ATTRIBUTIONS_FILENAME, index=False
        )
        record = {"technique": technique, "examples": {"digest": ce.examples_digest(examples)}}
        (folder / ce.RUN_FILENAME).write_text(json.dumps(record))
    return out_dir


@pytest.fixture
def outputs(tmp_path):
    return write_outputs(tmp_path / "out")


def edit_csv(path, change):
    frame = pd.read_csv(path, keep_default_na=False)
    change(frame)
    frame.to_csv(path, index=False)


class TestDrawing:
    def test_every_plot_and_page_is_written(self, outputs):
        plots = cp.run_plots(outputs, log=lambda *_: None)
        for technique in TECHNIQUES:
            assert (plots / "by_model" / technique / "beeswarm.png").stat().st_size > 0
            assert (plots / "models" / f"{technique}.html").exists()
            for n in (1, 2, 3):
                assert (plots / "by_model" / technique / f"waterfall_curated_0{n}.png").stat().st_size > 0
                assert (plots / "by_example" / f"curated_0{n}" / f"{technique}.png").exists()
        for n in (1, 2, 3):
            page = (plots / "examples" / f"curated_0{n}.html").read_text(encoding="utf-8")
            assert all(technique in page for technique in TECHNIQUES)
        index = (plots / "index.html").read_text(encoding="utf-8")
        assert all(f"curated_0{n}" in index for n in (1, 2, 3))
        assert index.count("/3</td>") == len(TECHNIQUES)

    def test_checks_are_text_free_and_record_the_gaps(self, outputs):
        plots = cp.run_plots(outputs, log=lambda *_: None)
        checks = pd.read_csv(plots / cp.CHECKS_FILENAME)
        assert len(checks) == len(TECHNIQUES) * len(TEXTS)
        assert not set(checks.columns) & ({"text", "token", "word"} | set(ce.TEXT_BEARING_COLUMNS))
        assert checks["additivity_gap"].abs().max() < 1e-9
        gaps = checks.set_index(["technique", "example_id"])["raw_vs_explained_gap"]
        assert gaps.xs("curated_03", level="example_id").to_numpy() == pytest.approx(0.02)

    def test_top_words_rank_by_total_absolute_attribution(self):
        contributions = pd.DataFrame(
            {
                "technique": ["T"] * 4,
                "example_id": ["a", "b", "a", "b"],
                "word": ["often", "often", "once", "rare"],
                "attribution": [0.1, -0.1, 0.15, 0.01],
            }
        )
        assert cp.top_words(contributions, "T", 2) == ["often", "once"]


class TestRefusals:
    def test_outputs_without_endpoints_are_refused(self, outputs):
        edit_csv(outputs / TECHNIQUES[0] / ce.PREDICTIONS_FILENAME,
                 lambda f: f.drop(columns=[ce.BASE_VALUE_COLUMN], inplace=True))
        with pytest.raises(cp.CuratedPlotsError, match="rerun"):
            cp.run_plots(outputs, log=lambda *_: None)

    def test_a_base_value_outside_zero_one_is_refused(self, outputs):
        edit_csv(outputs / TECHNIQUES[1] / ce.PREDICTIONS_FILENAME,
                 lambda f: f.__setitem__(ce.BASE_VALUE_COLUMN, -0.05))
        with pytest.raises(cp.CuratedPlotsError, match="not a probability"):
            cp.run_plots(outputs, log=lambda *_: None)

    def test_a_gap_in_token_positions_is_refused(self, outputs):
        path = outputs / TECHNIQUES[0] / ce.ATTRIBUTIONS_FILENAME
        frame = pd.read_csv(path, keep_default_na=False)
        frame = frame.drop(index=frame.index[(frame["example_id"] == "curated_01") & (frame["position"] == 1)])
        frame.to_csv(path, index=False)
        with pytest.raises(cp.CuratedPlotsError, match="token positions"):
            cp.run_plots(outputs, log=lambda *_: None)

    def test_attributions_that_do_not_add_up_are_refused(self, outputs):
        def nudge(frame):
            first = frame.index[frame["example_id"] == "curated_02"][0]
            frame.loc[first, "attribution"] += 0.05

        edit_csv(outputs / TECHNIQUES[1] / ce.ATTRIBUTIONS_FILENAME, nudge)
        with pytest.raises(cp.CuratedPlotsError, match="would not add up"):
            cp.run_plots(outputs, log=lambda *_: None)
        assert not (outputs / cp.PLOTS_DIRNAME / "index.html").exists()
