// grasp_mu_bisect.h
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// 数值稳定的 sigmoid
inline double grasp_stable_sigmoid(double x) {
	if (x >= 0.0) {
		double z = std::exp(-x);
		return 1.0 / (1.0 + z);
	} else {
		double z = std::exp(x);
		return z / (1.0 + z);
	}
}

// 计算期望边数 ∑_e σ((w_e + μ)/T)
inline double grasp_expected_edges_sum(const std::vector<double>& weights, double mu, double T) {
	double s = 0.0;
	for (double w : weights) s += grasp_stable_sigmoid((w + mu) / T);
	return s;
}

// 在温度 T 下通过二分寻找 μ，使期望边数接近 target
inline double grasp_find_mu_by_bisection(
		const std::vector<double>& weights,
		double T,
		double target,
		double tol = 1e-4,
		int max_iter = 60,
		double mu_min = std::numeric_limits<double>::quiet_NaN(),
		double mu_max = std::numeric_limits<double>::quiet_NaN()) {

	if (std::isnan(mu_min)) mu_min = -5.0 * T;
	if (std::isnan(mu_max)) mu_max =  5.0 * T;

	double lo = mu_min, hi = mu_max;
	for (int it = 0; it < max_iter && (hi - lo) > tol; ++it) {
		double mid = 0.5 * (lo + hi);
		double cur = grasp_expected_edges_sum(weights, mid, T);
		// 期望随 μ 单调增加：若当前期望小于目标，需增大 μ
		if (cur < target) lo = mid; else hi = mid;
	}
	return 0.5 * (lo + hi);
}

// 便捷封装：按“目标度 M*”求该节点的 μ
inline double grasp_find_mu_for_node_degree(
		const std::vector<double>& node_edge_weights,
		double T,
		double target_degree_M) {
	return grasp_find_mu_by_bisection(node_edge_weights, T, target_degree_M);
}


