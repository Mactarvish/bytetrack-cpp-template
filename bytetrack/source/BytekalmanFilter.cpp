#include "../include/BytekalmanFilter.h"
#include <Eigen/Cholesky>
#include <iostream>

// namespace byte_kalman
// {
	ByteKalmanFilter::ByteKalmanFilter()
	{
		int ndim = 4;
		double dt = 1.;
		// 假定了两帧的运动速度相同即匀速运动 xyah_k = xyah_k-1 + ·xyah_k-1
		_motion_mat = Eigen::MatrixXf::Identity(8, 8);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;
		}

		// 这个矩阵是Hk，用于把预测值的速度部分给砍掉
		_update_mat = Eigen::MatrixXf::Identity(4, 8);

		this->_std_weight_position = 1. / 20;
		this->_std_weight_velocity = 1. / 160;
	}

	KAL_DATA ByteKalmanFilter::initiate(const DETECTBOX &measurement) const
	{
		DETECTBOX mean_pos = measurement;
		DETECTBOX mean_vel;
		for (int i = 0; i < 4; i++) mean_vel(i) = 0;

		KAL_MEAN mean;
		for (int i = 0; i < 8; i++) {
			if (i < 4) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - 4);
		}

		KAL_MEAN std;
		std(0) = 2 * _std_weight_position * measurement[3];
		std(1) = 2 * _std_weight_position * measurement[3];
		std(2) = 1e-2;
		std(3) = 2 * _std_weight_position * measurement[3];
		std(4) = 10 * _std_weight_velocity * measurement[3];
		std(5) = 10 * _std_weight_velocity * measurement[3];
		std(6) = 1e-5;
		std(7) = 10 * _std_weight_velocity * measurement[3];

		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	// 预测公式的代码实现
	void ByteKalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) const
	{
		//revise the data;
		DETECTBOX std_pos;
		std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			1e-2,
			_std_weight_position * mean(3);
		DETECTBOX std_vel;
		std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			1e-5,
			_std_weight_velocity * mean(3);
		KAL_MEAN tmp;
		tmp.block<1, 4>(0, 0) = std_pos;
		tmp.block<1, 4>(0, 4) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	// Hkx~k 和 HkPkHk^T + Rk 的计算
	KAL_HDATA ByteKalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) const
	{
		DETECTBOX std;
		std << _std_weight_position * mean(3), _std_weight_position * mean(3),
			1e-1, _std_weight_position * mean(3);
		Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
		// 这个diag就是Rk
		diag = diag.array().square().matrix();
		// 只剩下了xyah的协方差矩阵，没有速度了，因为后边要计算观测值（即检测算法的输出）和预测值的差，而观测值没有速度，这个Hk的转换作用就是把预测值里的速度部分给砍掉
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		covariance1 += diag;
		// 只剩下了xyah的均值，没有速度了
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		return std::make_pair(mean1, covariance1);
	}

	KAL_DATA ByteKalmanFilter::update(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const DETECTBOX &measurement) const
	{
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;

		// B是covariance的前4排
		Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
		// 这个写法参考 https://zhuanlan.zhihu.com/p/589956755
		// 根据K'=PkHk^T(HkPkHk^T + Rk)^-1
		// (HkPkHk^T + Rk)K'=PkHk^T
		// 然后通过这个llt().solve()得到K'
		Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		// (zk - Hkxk~)T 注意这里有个转置
		Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
		auto tmp = innovation * (kalman_gain.transpose());
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

// }
