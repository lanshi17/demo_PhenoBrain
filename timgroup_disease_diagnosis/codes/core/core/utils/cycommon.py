import numpy as np


def to_rank_score(score_mat, arg_mat):
    eps = 1e-12
    row_num, col_num = score_mat.shape
    score_step = 1.0 / col_num

    for i in range(row_num):
        score = 0.0
        last_raw_score = score_mat[i, arg_mat[i, 0]]
        for j in range(1, col_num):
            col = arg_mat[i, j]
            diff = score_mat[i, col] - last_raw_score
            last_raw_score = score_mat[i, col]
            if not (-eps < diff < eps):
                score += score_step
            score_mat[i, col] = score
        score_mat[i, arg_mat[i, 0]] = 0.0
