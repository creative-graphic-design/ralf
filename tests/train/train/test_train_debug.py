import os
import sys

import pytest
from omegaconf import OmegaConf

from ralf.train.config import init_train_config_store


@pytest.mark.cuda
def test_train_debug_one_step() -> None:
    import pytest

    pytest.importorskip("cv2", exc_type=ImportError)
    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    data_dir = os.path.join(cache_root, "dataset", "cgl")

    init_train_config_store()
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "cgl",
                "data_dir": data_dir,
                "data_type": "parquet",
                "path": None,
                "max_seq_length": 10,
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "lr": 1e-4,
                "freeze_lr_epoch": 50,
                "freeze_dis_epoch": 50,
                "warmup_dis_epoch": 100,
                "plot_scalars_interval": 10,
                "plot_generated_samples_epoch_interval": 5,
                "log_level": "info",
                "save_tmp_model_epoch": 10000000,
                "save_vis_epoch": 100000,
                "clip_max_norm": 1.0,
                "num_workers": 0,
                "num_trainset": 2,
            },
            "data": {"transforms": ["image", "shuffle"], "tokenization": True},
            "sampling": {"name": "random", "temperature": 1.0},
            "tokenizer": {"num_bin": 128, "geo_quantization": "linear"},
            "generator": {
                "_target_": "ralf.train.models.autoreg.Autoreg",
                "_partial_": True,
                "d_model": 256,
            },
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "_partial_": True,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "_target_": "ralf.train.schedulers.VoidScheduler",
                "_partial_": True,
            },
            "discriminator": {"is_dummy": True},
            "job_dir": "/tmp/ralf_train_debug",
            "seed": 0,
            "debug": True,
            "use_ddp": False,
            "run_on_local": True,
        }
    )
    from ralf.train.train import main_worker

    main_worker(0, 1, cfg)


def test_filter_args_for_ai_platform_removes_job_dir_flags() -> None:
    from ralf.train import train

    old_argv = sys.argv
    sys.argv = ["prog", "--job_dir", "foo", "--job-dir=bar", "--other", "x"]
    try:
        train._filter_args_for_ai_platform()
        assert "--job_dir" not in sys.argv
        assert not any(arg.startswith("--job-dir") for arg in sys.argv)
    finally:
        sys.argv = old_argv
