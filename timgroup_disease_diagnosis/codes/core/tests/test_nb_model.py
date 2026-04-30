import warnings

import numpy as np

from core.predict.prob_model.nb_model import HPOProbMNBModel


class _Reader:
    def get_hpo_int_to_dis_int(self, phe_list_mode):
        return {0: [0, 1], 2: [1]}


class _Model:
    HPO_NUM = 4
    DIS_NUM = 2
    p2 = None
    hpo_reader = _Reader()


def test_background_log_prob_floors_absent_hpo_without_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = HPOProbMNBModel.get_background_log_prob_ary(_Model())

    assert [warning for warning in captured if issubclass(warning.category, RuntimeWarning)] == []
    np.testing.assert_allclose(result, [0.0, np.log(0.5), np.log(0.5), np.log(0.5)])
