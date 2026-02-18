from pathlib import Path

from ralf.compare.training_logs import prepare_migrated_training_logs


def test_prepare_migrated_training_logs(tmp_path: Path) -> None:
    src_root = tmp_path / "training_logs"
    src_root.mkdir()

    job_a = src_root / "job_a"
    job_a.mkdir()
    (job_a / "config.yaml").write_text(
        "generator:\n  _target_: image2layout.train.models.autoreg.Autoreg\n"
    )
    (job_a / "model.pt").write_text("checkpoint_a")

    job_b = src_root / "job_b"
    job_b.mkdir()
    (job_b / "config.yaml").write_text(
        "generator:\n  _target_: ralf.train.models.autoreg.Autoreg\n"
    )
    (job_b / "model.pt").write_text("checkpoint_b")

    dest_root = tmp_path / "training_logs_migrated"
    dest = prepare_migrated_training_logs(src_root, dest_root)

    assert dest == dest_root
    assert (dest_root / "job_a" / "model.pt").is_symlink()
    assert (dest_root / "job_b" / "model.pt").is_symlink()
    assert "image2layout" not in (dest_root / "job_a" / "config.yaml").read_text()
    assert "ralf" in (dest_root / "job_b" / "config.yaml").read_text()
