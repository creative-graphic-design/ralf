import os

from ralf.train.helpers.bucketizer import get_kmeans_cluster_center


def test_get_kmeans_cluster_center_realdata() -> None:
    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    weight_path = os.path.join(
        cache_root,
        "PRECOMPUTED_WEIGHT_DIR",
        "clustering",
        "cache",
        "cgl_kmeans_train_clusters.pkl",
    )
    centers = get_kmeans_cluster_center(weight_path=weight_path, key="center_x-128")
    assert centers.numel() == 128
