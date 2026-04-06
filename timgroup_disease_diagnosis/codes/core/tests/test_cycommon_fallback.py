import numpy as np

from core.utils.cycommon import to_rank_score


def test_to_rank_score_rewrites_matrix_in_place():
    score_mat = np.array([[0.1, 0.4, 0.4, 0.9]], dtype=np.float64)
    arg_mat = np.argsort(score_mat).astype(np.int32)

    to_rank_score(score_mat, arg_mat)

    assert np.allclose(score_mat, np.array([[0.0, 0.25, 0.25, 0.5]], dtype=np.float64))



def test_to_rank_score_handles_multiple_rows_independently():
    score_mat = np.array([
        [0.1, 0.4, 0.4, 0.9],
        [0.2, 0.3, 0.8, 0.8],
    ], dtype=np.float64)
    arg_mat = np.argsort(score_mat).astype(np.int32)

    to_rank_score(score_mat, arg_mat)

    assert np.allclose(score_mat[0], np.array([0.0, 0.25, 0.25, 0.5], dtype=np.float64))
    assert np.allclose(score_mat[1], np.array([0.0, 0.25, 0.5, 0.5], dtype=np.float64))
