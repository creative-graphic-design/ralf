import runpy
import sys
from pathlib import Path

from ralf.train.helpers import export_score_to_tex, export_score_to_tex_unanno


def _write_scores(path: Path, metrics: list[str], test_scores: list[str]) -> None:
    lines = ["header"]
    lines += metrics
    lines += [""] * 4
    lines += test_scores
    lines += [""] * 5
    path.write_text("\n".join(lines))


def test_export_score_to_tex(tmp_path: Path) -> None:
    root = tmp_path / "results" / "exp"
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "generated_samples_uncond_dummy"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "alignment-LayoutGAN++",
        "overlap-LayoutGAN++",
        "utilization",
        "occlusion",
        "unreadability",
        "overlay",
        "underlay_effectiveness_loose",
        "underlay_effectiveness_strict",
        "R_{shm} (vgg distance)",
        "validity",
        "test_precision_layout",
        "test_recall_layout",
        "test_density_layout",
        "test_coverage_layout",
        "test_fid_layout",
    ]
    test_scores = ["0.1"] * len(metrics)
    _write_scores(out_dir / "scores_all.txt", metrics, test_scores)

    scores = export_score_to_tex.load_k_scores(str(root))
    assert "uncond" in scores
    export_score_to_tex.export_score_as_csv(str(root))
    assert (root / "scores_test.tex").exists()


def test_export_score_to_tex_unanno(tmp_path: Path) -> None:
    root = tmp_path / "results" / "exp"
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "no_anno_data_uncond_dummy"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "alignment-LayoutGAN++",
        "overlap-LayoutGAN++",
        "utilization",
        "occlusion",
        "unreadability",
        "overlay",
        "underlay_effectiveness_loose",
        "underlay_effectiveness_strict",
        "R_{shm} (vgg distance)",
        "validity",
    ]
    test_scores = ["0.2"] * len(metrics)
    _write_scores(out_dir / "scores_all.txt", metrics, test_scores)

    scores = export_score_to_tex_unanno.load_k_scores(str(root))
    assert "uncond" in scores
    export_score_to_tex_unanno.export_score_as_csv(str(root))
    assert (root / "scores_test_unanno.tex").exists()


def test_export_score_to_tex_branches(tmp_path: Path) -> None:
    root = tmp_path / "results" / "exp"
    root.mkdir(parents=True, exist_ok=True)

    metrics = [
        "alignment-LayoutGAN++",
        "overlap-LayoutGAN++",
        "utilization",
        "occlusion",
        "unreadability",
        "overlay",
        "underlay_effectiveness_loose",
        "underlay_effectiveness_strict",
        "R_{shm} (vgg distance)",
        "validity",
        "test_precision_layout",
        "test_recall_layout",
        "test_density_layout",
        "test_coverage_layout",
        "test_fid_layout",
    ]
    test_scores = ["0.1"] * len(metrics)

    dynamic_dir = root / "generated_samples_uncond_dummy_dynamictopk_4"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(dynamic_dir / "scores_all.txt", metrics, test_scores)

    backtrack_dir = root / "generated_samples_relation_dummy_backtrack"
    backtrack_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(backtrack_dir / "scores_all.txt", metrics, test_scores)

    c_dir = root / "generated_samples_c_dummy"
    c_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(c_dir / "scores_all.txt", metrics, test_scores)

    skip_dir = root / "generated_samples_unknown_dummy"
    skip_dir.mkdir(parents=True, exist_ok=True)

    scores = export_score_to_tex.load_k_scores(str(root))
    assert "relation_backtrack" in scores
    assert "4" in scores["uncond"]

    export_score_to_tex.export_score_as_csv(str(root))
    assert (root / "scores_test.tex").exists()

    old_argv = sys.argv
    sys.argv = ["export_score_to_tex", "--root", str(root)]
    try:
        runpy.run_module("ralf.train.helpers.export_score_to_tex", run_name="__main__")
    finally:
        sys.argv = old_argv


def test_export_score_to_tex_unanno_branches(tmp_path: Path) -> None:
    root = tmp_path / "results" / "exp"
    root.mkdir(parents=True, exist_ok=True)

    metrics = [
        "alignment-LayoutGAN++",
        "overlap-LayoutGAN++",
        "utilization",
        "occlusion",
        "unreadability",
        "overlay",
        "underlay_effectiveness_loose",
        "underlay_effectiveness_strict",
        "R_{shm} (vgg distance)",
        "validity",
    ]
    test_scores = ["0.2"] * len(metrics)

    dynamic_dir = root / "no_anno_data_uncond_dummy_dynamictopk_2"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(dynamic_dir / "scores_all.txt", metrics, test_scores)

    skip_dir = root / "no_anno_data_unknown_dummy"
    skip_dir.mkdir(parents=True, exist_ok=True)

    scores = export_score_to_tex_unanno.load_k_scores(str(root))
    assert "uncond" in scores

    export_score_to_tex_unanno.export_score_as_csv(str(root))
    assert (root / "scores_test_unanno.tex").exists()

    old_argv = sys.argv
    sys.argv = ["export_score_to_tex_unanno", "--root", str(root)]
    try:
        runpy.run_module(
            "ralf.train.helpers.export_score_to_tex_unanno", run_name="__main__"
        )
    finally:
        sys.argv = old_argv
