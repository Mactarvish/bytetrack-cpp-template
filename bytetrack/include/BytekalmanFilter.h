#pragma once

#include "dataType.h"

#ifdef WIN32
#ifdef BYTETRACK_LIBRARY_EXPORTS
#define BYTETRACK_LIBRARY_API __declspec(dllexport)
#else
#define BYTETRACK_LIBRARY_API __declspec(dllimport)
#endif
#else
#define BYTETRACK_LIBRARY_API
#endif


// namespace byte_kalman
// {
	class ByteKalmanFilter
	{
	public:
		static const double chi2inv95[10];
		BYTETRACK_LIBRARY_API ByteKalmanFilter();
		BYTETRACK_LIBRARY_API void predict(KAL_MEAN& mean, KAL_COVA& covariance) const;
		BYTETRACK_LIBRARY_API KAL_DATA initiate(const DETECTBOX& measurement) const;
		BYTETRACK_LIBRARY_API KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance) const;
		BYTETRACK_LIBRARY_API KAL_DATA update(const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const DETECTBOX& measurement) const;

	private:
		Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
		Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
		float _std_weight_position;
		float _std_weight_velocity;
	};
// }