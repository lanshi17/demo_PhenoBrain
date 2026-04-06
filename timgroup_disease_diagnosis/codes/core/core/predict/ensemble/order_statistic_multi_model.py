import heapq
import math
import numpy as np
from copy import deepcopy

from core.predict.model import Model
from core.reader.hpo_reader import HPOReader


def score_vec_to_rank_ratios(score_vec):
	score_vec = np.asarray(score_vec, dtype=np.float64)
	dis_num = score_vec.shape[0]
	order = np.argsort(-score_vec, kind='mergesort')
	ranks = np.empty(dis_num, dtype=np.float64)
	current_rank = 1
	previous_score = score_vec[order[0]]
	ranks[order[0]] = current_rank
	for position in range(1, dis_num):
		col = order[position]
		score = score_vec[col]
		if not math.isclose(score, previous_score, rel_tol=0.0, abs_tol=1e-12):
			current_rank = position + 1
			previous_score = score
		ranks[col] = current_rank
	return ranks / dis_num


def stuart_order_statistic_pvalue(sorted_rank_ratios):
	sorted_rank_ratios = np.asarray(sorted_rank_ratios, dtype=np.float64)
	model_num = sorted_rank_ratios.shape[0]
	v = np.zeros(model_num + 1, dtype=np.float64)
	v[0] = 1.0
	for k in range(1, model_num + 1):
		ratio = sorted_rank_ratios[model_num - k]
		total = 0.0
		for i in range(1, k + 1):
			total += ((-1) ** (i - 1)) * v[k - i] * (ratio ** i) / math.factorial(i)
		v[k] = total
	z = math.factorial(model_num) * v[model_num]
	return min(max(z, np.finfo(np.float64).tiny), 1.0)


def stuart_order_statistic_score(sorted_rank_ratios):
	return -math.log(stuart_order_statistic_pvalue(sorted_rank_ratios))


def combine_score_vecs_with_order_statistics(score_vecs):
	rank_ratio_mat = np.vstack([score_vec_to_rank_ratios(score_vec) for score_vec in score_vecs])
	fused_score_vec = np.zeros(rank_ratio_mat.shape[1], dtype=np.float64)
	for col in range(rank_ratio_mat.shape[1]):
		fused_score_vec[col] = stuart_order_statistic_score(np.sort(rank_ratio_mat[:, col]))
	return fused_score_vec


class OrderStatisticMultiModel(Model):
	def __init__(self, model_inits=None, hpo_reader=HPOReader(), model_name=None, model_list=None, keep_raw_score=True):
		super(OrderStatisticMultiModel, self).__init__()
		self.name = model_name or 'OrderStatisticMultiModel'
		self.model_list = model_list or [init_func(*init_args, **init_kwargs) for init_func, init_args, init_kwargs in model_inits]
		self.keep_raw_score = keep_raw_score

		self.hpo_reader = hpo_reader
		self.DIS_NUM = hpo_reader.get_dis_num()
		self.dis_list = hpo_reader.get_dis_list()


	def query_score_vec(self, phe_list):
		if len(phe_list) == 0:
			return self.query_empty_score_vec()
		raw_score_vecs = [model.query_score_vec(phe_list) for model in self.model_list]
		self.raw_score_mats = [np.reshape(score_vec, (1, -1)) for score_vec in raw_score_vecs]
		return self.combine_score_vecs(raw_score_vecs)


	def combine_score_vecs(self, score_vecs):
		return combine_score_vecs_with_order_statistics(score_vecs)


	def query_score_mat(self, phe_lists, chunk_size=200, cpu_use=12):
		raw_score_mats = [model.query_score_mat(phe_lists, chunk_size=chunk_size, cpu_use=cpu_use) for model in self.model_list]
		if self.keep_raw_score:
			self.raw_score_mats = deepcopy(raw_score_mats)
		ret_mat = []
		for i in range(len(phe_lists)):
			ret_mat.append(self.combine_score_vecs([score_mat[i] for score_mat in raw_score_mats]))
		return np.vstack(ret_mat)


	def score_vec_to_result(self, score_vec, topk, pa_idx=0):
		if self.keep_raw_score:
			if topk is None:
				dis_int_scores = sorted([(i, score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1], reverse=True)
			else:
				dis_int_scores = heapq.nlargest(topk, [(i, score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1])
			return [(self.dis_list[i], self.raw_score_mats[0][pa_idx][i]) for i, _ in dis_int_scores]
		if topk is None:
			return sorted([(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1], reverse=True)
		return heapq.nlargest(topk, [(self.dis_list[i], score_vec[i]) for i in range(self.DIS_NUM)], key=lambda item: item[1])


	def score_mat_to_results(self, phe_lists, score_mat, topk):
		return [
			self.score_vec_to_result(score_mat[i], topk, pa_idx=i) if len(phe_lists[i]) != 0 else self.query_empty(topk)
			for i in range(score_mat.shape[0])
		]
