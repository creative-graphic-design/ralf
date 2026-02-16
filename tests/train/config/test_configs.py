from ralf.train.config import (
    dataset_config_factory,
    dataset_config_names,
    sampling_config_factory,
    sampling_config_names,
)


def test_dataset_config_factory() -> None:
    for name in dataset_config_names():
        cfg = dataset_config_factory(name)()
        assert hasattr(cfg, "name")


def test_sampling_config_factory() -> None:
    for name in sampling_config_names():
        cfg = sampling_config_factory(name)()
        assert hasattr(cfg, "name")
