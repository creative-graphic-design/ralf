import os
import runpy

import numpy as np

from ralf.train.models.retrieval.reranker import (
    maximal_marginal_relevance,
    reranker_random,
    reranker_top_k,
)


def test_reranker_functions() -> None:
    score_di_q = np.array([0.1, 0.2, 0.3])
    score_di_dj = np.eye(3)
    mmr = maximal_marginal_relevance(
        score_di_q=score_di_q,
        score_di_dj=score_di_dj,
        lam=0.5,
        top_k=2,
        score_type="similarity",
    )
    assert len(mmr) == 2
    topk = reranker_top_k(score_di_q, top_k=2, score_type="similarity")
    assert len(topk) == 2
    rnd = reranker_random(5, top_k=2)
    assert len(rnd) == 2


def test_reranker_distance_and_error_branches() -> None:
    score_di_q = np.array([0.3, 0.1, 0.2])
    score_di_dj = np.eye(3)
    order = maximal_marginal_relevance(
        score_di_q=score_di_q,
        score_di_dj=score_di_dj,
        lam=0.2,
        top_k=2,
        score_type="distance",
    )
    assert order[0] == 1

    try:
        maximal_marginal_relevance(
            score_di_q=score_di_q,
            score_di_dj=score_di_dj,
            lam=0.5,
            top_k=2,
            score_type="invalid",
        )
        assert False, "Expected ValueError for invalid score_type"
    except ValueError:
        pass

    try:
        maximal_marginal_relevance(
            score_di_q=score_di_q,
            score_di_dj=score_di_dj,
            lam=0.5,
            top_k=0,
            score_type="similarity",
        )
        assert False, "Expected ValueError for invalid top_k"
    except ValueError:
        pass

    try:
        maximal_marginal_relevance(
            score_di_q=score_di_q,
            score_di_dj=score_di_dj,
            lam=1.5,
            top_k=1,
            score_type="similarity",
        )
        assert False, "Expected ValueError for invalid lambda"
    except ValueError:
        pass


def test_reranker_main_creates_plot(tmp_path) -> None:
    cwd = os.getcwd()
    os.environ["MPLBACKEND"] = "Agg"
    old_argv = os.sys.argv
    os.sys.argv = ["reranker"]
    os.chdir(tmp_path)
    try:
        runpy.run_module("ralf.train.models.retrieval.reranker", run_name="__main__")
        assert (tmp_path / "comparison_retrieval.png").exists()
    finally:
        os.chdir(cwd)
        os.sys.argv = old_argv
