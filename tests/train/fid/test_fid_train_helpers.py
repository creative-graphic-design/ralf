from ralf.train.fid.train import save_checkpoint


def test_save_checkpoint(tmp_path) -> None:
    state = {"epoch": 1}
    save_checkpoint(state, is_best=True, out_dir=tmp_path)
    assert (tmp_path / "checkpoint.pth.tar").exists()
    assert (tmp_path / "model_best.pth.tar").exists()
