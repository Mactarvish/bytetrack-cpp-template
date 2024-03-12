#pragma once

#include "BytekalmanFilter.h"

enum TrackState { New = 0, Tracked, Lost, Removed };

template<typename DetectionResult>
class STrack
{
public:
	STrack( std::vector<float> tlwh_, int category, float score);
	~STrack();

	std::vector<float> static tlbr_to_tlwh( std::vector<float> &tlbr);
	// void static multi_predict( std::vector<STrack*> &stracks, byte_kalman::ByteKalmanFilter &kalman_filter);
	void static multi_predict( std::vector<STrack*> &stracks, ByteKalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	std::vector<float> tlwh_to_xyah( std::vector<float> tlwh_tmp);
	std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	// void activate(byte_kalman::ByteKalmanFilter &kalman_filter, int frame_id);
	void activate(ByteKalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

public:
	bool is_activated;
	int track_id;
	int state;

	std::vector<float> _tlwh;
	std::vector<float> tlwh;
	std::vector<float> tlbr;
	DetectionResult detection;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;
	int category;

private:
	ByteKalmanFilter kalman_filter;
};

template<typename DetectionResult>
STrack<DetectionResult>::STrack(std::vector<float> tlwh_, int category, float score)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;
	track_id = 0;
	state = TrackState::New;

	tlwh.resize(4);
	tlbr.resize(4);

	static_tlwh();
	static_tlbr();
	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	this->category = category;
	start_frame = 0;

	this->detection = DetectionResult(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]), category, score, 0);
}

template<typename DetectionResult>
STrack<DetectionResult>::~STrack()
{
}

template<typename DetectionResult>
void STrack<DetectionResult>::activate(ByteKalmanFilter& kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	std::vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	// if (frame_id == 1)
	// {
	// 	this->is_activated = true;
	// }
	this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

template<typename DetectionResult>
void STrack<DetectionResult>::re_activate(STrack& new_track, int frame_id, bool new_id)
{
	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id)
		this->track_id = next_id();

	auto category = 0;
	this->detection = DetectionResult(int(tlwh[0]), int(tlwh[1]), int(tlwh[0]) + int(tlwh[2]), int(tlwh[1]) + int(tlwh[3]), category, new_track.score, this->track_id);
}

template<typename DetectionResult>
void STrack<DetectionResult>::update(STrack& new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->state = TrackState::Tracked;
	this->is_activated = true;

	this->score = new_track.score;
	auto category = 0;
	this->detection = DetectionResult(int(tlwh[0]), int(tlwh[1]), int(tlwh[0]) + int(tlwh[2]), int(tlwh[1]) + int(tlwh[3]), category, new_track.score, this->track_id);
}

template<typename DetectionResult>
void STrack<DetectionResult>::static_tlwh()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	// xyah -> tlwh
	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];
	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

template<typename DetectionResult>
void STrack<DetectionResult>::static_tlbr()
{
	tlbr = tlwh;
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

template<typename DetectionResult>
std::vector<float> STrack<DetectionResult>::tlwh_to_xyah(std::vector<float> tlwh_tmp)
{
	std::vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

template<typename DetectionResult>
std::vector<float> STrack<DetectionResult>::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

template<typename DetectionResult>
std::vector<float> STrack<DetectionResult>::tlbr_to_tlwh(std::vector<float>& tlbr)
{
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

template<typename DetectionResult>
void STrack<DetectionResult>::mark_lost()
{
	state = TrackState::Lost;
}

template<typename DetectionResult>
void STrack<DetectionResult>::mark_removed()
{
	state = TrackState::Removed;
}

template<typename DetectionResult>
int STrack<DetectionResult>::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

template<typename DetectionResult>
int STrack<DetectionResult>::end_frame()
{
	return this->frame_id;
}

template<typename DetectionResult>
void STrack<DetectionResult>::multi_predict(std::vector<STrack*>& stracks, ByteKalmanFilter& kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		if (stracks[i]->state != TrackState::Tracked)
		{
			stracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
		stracks[i]->static_tlwh();
		stracks[i]->static_tlbr();
	}
}