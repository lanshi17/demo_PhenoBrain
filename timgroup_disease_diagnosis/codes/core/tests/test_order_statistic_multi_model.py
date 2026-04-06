import numpy as np

from core.predict.ensemble.order_statistic_multi_model import (
    combine_score_vecs_with_order_statistics,
    score_vec_to_rank_ratios,
    stuart_order_statistic_score,
)


def test_score_vec_to_rank_ratios_uses_descending_ranks():
    score_vec = np.array([0.9, 0.1, 0.5], dtype=np.float64)

    rank_ratios = score_vec_to_rank_ratios(score_vec)

    assert np.allclose(rank_ratios, np.array([1 / 3, 1.0, 2 / 3], dtype=np.float64))


def test_stuart_order_statistic_score_prefers_consensus_of_excellence():
    better = stuart_order_statistic_score(np.array([0.01, 0.02, 0.05], dtype=np.float64))
    worse = stuart_order_statistic_score(np.array([0.2, 0.2, 0.2], dtype=np.float64))

    assert better > worse


def test_combine_score_vecs_with_order_statistics_prefers_top_consensus():
    score_vecs = [
        np.array([0.99, 0.60, 0.10], dtype=np.float64),
        np.array([0.98, 0.20, 0.60], dtype=np.float64),
        np.array([0.97, 0.30, 0.50], dtype=np.float64),
    ]

    fused = combine_score_vecs_with_order_statistics(score_vecs)

    assert fused[0] > fused[1]
    assert fused[0] > fused[2]
