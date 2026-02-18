from pathlib import Path

from ralf.compare.job_dir import prepare_migrated_job_dir


def test_prepare_migrated_job_dir_rewrites_config(tmp_path: Path) -> None:
    source = tmp_path / "job"
    source.mkdir()
    (source / "config.yaml").write_text(
        "generator:\n  _target_: image2layout.train.models.autoreg.Autoreg\n"
    )
    (source / "model.pt").write_text("checkpoint")

    dest_root = tmp_path / "dest"
    migrated = prepare_migrated_job_dir(source, dest_root)

    assert migrated != source
    assert migrated.exists()
    assert "image2layout" not in (migrated / "config.yaml").read_text()
    assert "ralf" in (migrated / "config.yaml").read_text()
    assert (migrated / "model.pt").is_symlink()
    assert (migrated / "model.pt").resolve() == (source / "model.pt").resolve()


def test_prepare_migrated_job_dir_no_rewrite(tmp_path: Path) -> None:
    source = tmp_path / "job_no_rewrite"
    source.mkdir()
    (source / "config.yaml").write_text(
        "generator:\n  _target_: ralf.train.models.autoreg.Autoreg\n"
    )

    dest_root = tmp_path / "dest_no_rewrite"
    migrated = prepare_migrated_job_dir(source, dest_root)

    assert migrated == source
    assert not dest_root.exists()


def test_prepare_migrated_job_dir_force_mirror(tmp_path: Path) -> None:
    source = tmp_path / "job_force"
    source.mkdir()
    (source / "config.yaml").write_text(
        "generator:\n  _target_: ralf.train.models.autoreg.Autoreg\n"
    )
    (source / "model.pt").write_text("checkpoint")

    dest_root = tmp_path / "dest_force"
    migrated = prepare_migrated_job_dir(source, dest_root, force_mirror=True)

    assert migrated != source
    assert (migrated / "config.yaml").read_text().startswith("generator")
    assert (migrated / "model.pt").is_symlink()
