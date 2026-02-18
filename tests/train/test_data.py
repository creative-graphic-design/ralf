from ralf.train.helpers.util import argsort


def test_argsort() -> None:
    input_, expected = [0.3, 1.0, 0.5], [0, 2, 1]
    assert argsort(input_) == expected


def test_get_dataset_with_env(dataset_dir) -> None:
    from ralf.train.data import get_dataset

    cfg = type("Cfg", (), {"data_dir": dataset_dir, "path": "", "data_type": "parquet"})
    dataset, features = get_dataset(dataset_cfg=cfg, transforms=["image"])
    assert "train" in dataset
    assert "label" in features


if __name__ == "__main__":
    test_argsort()
