/*
* https://www.cnblogs.com/jiayouwyhit/p/3836547.html
* 模板类的编写，不能将声明和定义分开，否则会出现链接错误
* 将template function 或者 template class的完整定义直接放在.h文件中，然后加到要使用这些template function的.cpp文件中。
*/

#pragma once

#include "STrack.h"
#include "lapjv.h"

template<typename DetectionResult>
class BYTETracker
{
public:
	BYTETracker(float track_thresh, float high_thresh, float match_thresh, int max_time_lost, int time_lost_pred);
	~BYTETracker();

	//std::vector<STrack<DetectionResult>> update(const  std::vector<DetectionResult>& objects);
	std::vector<DetectionResult> update(const  std::vector<DetectionResult>& objects);
    cv::Scalar get_color(int idx);

private:
	std::vector<STrack<DetectionResult>*> joint_stracks( std::vector<STrack<DetectionResult>*> &tlista,  std::vector<STrack<DetectionResult>> &tlistb);
	std::vector<STrack<DetectionResult>> joint_stracks( std::vector<STrack<DetectionResult>> &tlista,  std::vector<STrack<DetectionResult>> &tlistb);

	std::vector<STrack<DetectionResult>> sub_stracks( std::vector<STrack<DetectionResult>> &tlista,  std::vector<STrack<DetectionResult>> &tlistb);
	void remove_duplicate_stracks( std::vector<STrack<DetectionResult>> &resa,  std::vector<STrack<DetectionResult>> &resb,  std::vector<STrack<DetectionResult>> &stracksa,  std::vector<STrack<DetectionResult>> &stracksb);

	void linear_assignment( std::vector< std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		 std::vector< std::vector<int> > &matches,  std::vector<int> &unmatched_a,  std::vector<int> &unmatched_b);
	std::vector< std::vector<float> > iou_distance( std::vector<STrack<DetectionResult>> &atracks,  std::vector<STrack<DetectionResult>> &btracks);
	std::vector< std::vector<float> > iou_distance( std::vector<STrack<DetectionResult>*> &atracks,  std::vector<STrack<DetectionResult>> &btracks);
	std::vector< std::vector<float> > ious( std::vector< std::vector<float> > &atlbrs,  std::vector< std::vector<float> > &btlbrs);

	double lapjv(const  std::vector< std::vector<float> > &cost,  std::vector<int> &rowsol,  std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;
	int time_lost_pred;

	std::vector<STrack<DetectionResult>> tracked_stracks;
	std::vector<STrack<DetectionResult>> lost_stracks;
	std::vector<STrack<DetectionResult>> removed_stracks;
	ByteKalmanFilter kalman_filter;
};

template<typename DetectionResult>
BYTETracker<DetectionResult>::BYTETracker(float track_thresh, float high_thresh, float match_thresh, int max_time_lost, int time_lost_pred)
{
	this->track_thresh = track_thresh;
	this->high_thresh = high_thresh;
	this->match_thresh = match_thresh;
	this->max_time_lost = max_time_lost;
	this->time_lost_pred = time_lost_pred;

	frame_id = 0;
}

template<typename DetectionResult>
BYTETracker<DetectionResult>::~BYTETracker()
{
}

template<typename DetectionResult>
std::vector<DetectionResult> BYTETracker<DetectionResult>::update(const std::vector<DetectionResult>& objects)
{
	////////////////// Step 1: Get detections_high //////////////////
	this->frame_id++;
	std::vector<STrack<DetectionResult>> dets_high{};
	std::vector<STrack<DetectionResult>> dets_low{};
	// 每个检测框构造一个跟踪器，分为高分段和低分段
	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(4);
            auto [x, y, w, h] = objects[i].GetXYWH();
            tlbr_[0] = x;
            tlbr_[1] = y;
			tlbr_[2] = x + w;
            tlbr_[3] = y + h;

			float score = objects[i].confidence;
			int category = objects[i].category;

			STrack<DetectionResult> strack(STrack<DetectionResult>::tlbr_to_tlwh(tlbr_), category, score);
			if (score >= track_thresh)
			{
				dets_high.push_back(strack);
			}
			else
			{
				dets_low.push_back(strack);
			}
		}
	}

	// 管理现有的跟踪器，处于未激活状态的划分为“不确定组”，否则划分为“正在跟踪组”
	std::vector<STrack<DetectionResult>*> unconfirmed_stracks{};
	std::vector<STrack<DetectionResult>*> tracked_stracks{};
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			tracked_stracks.push_back(&this->tracked_stracks[i]);
		}
		else
		{
			unconfirmed_stracks.push_back(&this->tracked_stracks[i]);
		}
	}

	////////////////// Step 2: First association, with IoU //////////////////
	// 跟踪中激活态的跟踪器和丢失态的跟踪器取并集，相同id的以前者为准
	std::vector<STrack<DetectionResult>*> strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	// ! 逐个跟踪器进行预测，整个算法的核心就在这里
	STrack<DetectionResult>::multi_predict(strack_pool, this->kalman_filter);

	// 正在跟踪组的跟踪器和高分段检测框进行匹配
	std::vector< std::vector<float> > dists = iou_distance(strack_pool, dets_high);
	std::vector< std::vector<int> > matches{};
	std::vector<int> u_track_inds{};
	std::vector<int> u_det_inds{};
	linear_assignment(dists, strack_pool.size(), dets_high.size(), match_thresh, matches, u_track_inds, u_det_inds);

	// 根据匹配结果更新跟踪器的状态，对于原本就处于跟踪中的跟踪器，保持原样；对于之前失活的跟踪器，重新激活并划分到“重新找到组”
	std::vector<STrack<DetectionResult>> activated_stracks{};
	std::vector<STrack<DetectionResult>> refind_stracks{};
	for (int i = 0; i < matches.size(); i++)
	{
		STrack<DetectionResult>* track = strack_pool[matches[i][0]];
		STrack<DetectionResult>* det = &dets_high[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	// 经过上述匹配，高分段检测框有一些有主了，另一些没有被现有的跟踪器匹配到，这些失配框我们先保存起来放到一边，先处理上文中那些低分段的检测框
	std::vector<STrack<DetectionResult>> u_dets_high{};
	for (int i = 0; i < u_det_inds.size(); i++)
	{
		u_dets_high.push_back(dets_high[u_det_inds[i]]);
	}
	u_det_inds.clear();

	// 接下来我们处理没有匹配到框的跟踪器，先把之前处于跟踪中的跟踪器划分为一组
	std::vector<STrack<DetectionResult>*> r_tracked_stracks;
	for (int i = 0; i < u_track_inds.size(); i++)
	{
		if (strack_pool[u_track_inds[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track_inds[i]]);
		}
	}
	// 计算这些暂时失势的跟踪器和低分段检测框的匹配
	dists = iou_distance(r_tracked_stracks, dets_low);
	matches.clear();
	u_track_inds.clear();
	linear_assignment(dists, r_tracked_stracks.size(), dets_low.size(), match_thresh, matches, u_track_inds, u_det_inds);
	// 重新激活失势跟踪器
	for (int i = 0; i < matches.size(); i++)
	{
		STrack<DetectionResult>* track = r_tracked_stracks[matches[i][0]];
		STrack<DetectionResult>* det = &dets_low[matches[i][1]];
		track->update(*det, this->frame_id);
		activated_stracks.push_back(*track);
	}

	// 这次还没匹配到框的跟踪器，那就只能标记为丢失了
	std::vector<STrack<DetectionResult>> lost_stracks{};
	for (int i = 0; i < u_track_inds.size(); i++)
	{
		STrack<DetectionResult>* track = r_tracked_stracks[u_track_inds[i]];
		track->mark_lost();
		lost_stracks.push_back(*track);
	}

	// Deal with unconfirmed_stracks tracks, usually tracks with only one beginning frame
	// 现在处理高分段失配框，这些框和最开始的不确定跟踪器进行匹配
	dists = iou_distance(unconfirmed_stracks, u_dets_high);

	matches.clear();
	u_det_inds.clear();
	std::vector<int> u_unconfirmed{};
	linear_assignment(dists, unconfirmed_stracks.size(), u_dets_high.size(), match_thresh, matches, u_unconfirmed, u_det_inds);

	// 匹配到的标记为激活，未匹配到的标记为删除
	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed_stracks[matches[i][0]]->update(u_dets_high[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed_stracks[matches[i][0]]);
	}
	std::vector<STrack<DetectionResult>> removed_stracks;
	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack<DetectionResult>* track = unconfirmed_stracks[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	// 处理剩余的高分段失配框
	for (int i = 0; i < u_det_inds.size(); i++)
	{
		STrack<DetectionResult>* track = &u_dets_high[u_det_inds[i]];
		// 划分高低分段检测框的时候用的是track_thresh，这里是high_thresh，高于high_thresh的才会被激活
		if (track->score >= this->high_thresh)
		{
			track->activate(this->kalman_filter, this->frame_id);
			activated_stracks.push_back(*track);
		}
	}

	////////////////// Step 5: Update state //////////////////
	// 处理丢失跟踪器，如果丢失的时间足够长了，那么标记为删除
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}

	// 净化this->tracked_stracks，只保留处于跟踪状态的跟踪器
	int t = 0;
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			this->tracked_stracks[t] = this->tracked_stracks[i];
			t++;
		}
	}
	this->tracked_stracks.resize(t, { {0,0,0,0}, 0, 0 });
	// 跟踪中、激活状态、重找到的跟踪器取并集，相同id的以前者为准
	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);
	// 从丢失态跟踪器集合中删除那些对应id已经有跟踪中态跟踪器占用的跟踪器
	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}
	// 同上，择出具有和已经被删除的跟踪器所跟踪的id相同的id的丢失态跟踪器
	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}

	// 两两计算跟踪态跟踪器和丢失态跟踪器的iou，如果大于一定程度，表明二者其一是冗余的，这时删除其中岁数更小的那个
	std::vector<STrack<DetectionResult>> resa{};
	std::vector<STrack<DetectionResult>> resb{};
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);
	this->tracked_stracks = resa;
	this->lost_stracks = resb;

	// 跟踪中的跟踪器转输出
	std::vector<DetectionResult> output_stracks{};
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			std::vector<float> tlwh = this->tracked_stracks[i].tlwh;
			float conf = this->tracked_stracks[i].score;
			int track_id = this->tracked_stracks[i].track_id;
			auto category = this->tracked_stracks[i].category;
            DetectionResult temp = DetectionResult(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]), category, conf, track_id);
			output_stracks.push_back(temp);
		}
	}
	// 丢失状态的跟踪器如果丢的时间不超过阈值那么也输出
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() <= this->time_lost_pred)
		{
			std::vector<float> tlwh = this->lost_stracks[i].tlwh;
			float conf = this->lost_stracks[i].score;
			int track_id = this->lost_stracks[i].track_id;
            auto category = this->lost_stracks[i].category;
            DetectionResult temp = DetectionResult(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]), category, conf, track_id);
			output_stracks.push_back(temp);
		}
	}

	return output_stracks;
}

template<typename DetectionResult>
std::vector<STrack<DetectionResult>*> BYTETracker<DetectionResult>::joint_stracks(std::vector<STrack<DetectionResult>*>& tlista, std::vector<STrack<DetectionResult>>& tlistb)
{
    std::set<int> exists;
    std::vector<STrack<DetectionResult>*> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(tlista[i]->track_id);
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (!exists.count(tid))
        {
            exists.insert(tid);
            res.push_back(&tlistb[i]);
        }
    }
    return res;
}

template<typename DetectionResult>
std::vector<STrack<DetectionResult>> BYTETracker<DetectionResult>::joint_stracks(std::vector<STrack<DetectionResult>>& tlista, std::vector<STrack<DetectionResult>>& tlistb)
{
    std::set<int> exists;
    std::vector<STrack<DetectionResult>> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(tlista[i].track_id);
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (!exists.count(tid))
        {
            exists.insert(tid);
            res.push_back(tlistb[i]);
        }
    }
    return res;
}

template<typename DetectionResult>
std::vector<STrack<DetectionResult>> BYTETracker<DetectionResult>::sub_stracks(std::vector<STrack<DetectionResult>>& tlista, std::vector<STrack<DetectionResult>>& tlistb)
{
    std::map<int, STrack<DetectionResult>> stracks;
    for (int i = 0; i < tlista.size(); i++)
    {
        stracks.insert(std::pair<int, STrack<DetectionResult>>(tlista[i].track_id, tlista[i]));
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid))
        {
            stracks.erase(tid);
        }
    }

    std::vector<STrack<DetectionResult>> res;
    // 注意交叉编译安卓的时候要求加这么个typename
    for (typename std::map<int, STrack<DetectionResult>>::iterator it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

template<typename DetectionResult>
void BYTETracker<DetectionResult>::remove_duplicate_stracks(std::vector<STrack<DetectionResult>>& resa, std::vector<STrack<DetectionResult>>& resb, std::vector<STrack<DetectionResult>>& stracksa, std::vector<STrack<DetectionResult>>& stracksb)
{
    std::vector< std::vector<float> > pdist = iou_distance(stracksa, stracksb);
    std::vector<std::pair<int, int> > pairs;
    for (int i = 0; i < pdist.size(); i++)
    {
        for (int j = 0; j < pdist[i].size(); j++)
        {
            if (pdist[i][j] < 0.15)
            {
                pairs.push_back(std::pair<int, int>(i, j));
            }
        }
    }
    // 对于两个十分接近的框，保留那个时间长的
    // 注意这个dup保存的是冗余的也就是亟待删除的那个框
    std::vector<int> dupa, dupb;
    for (int i = 0; i < pairs.size(); i++)
    {
        int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
        int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
        if (timep > timeq)
            dupb.push_back(pairs[i].second);
        else
            dupa.push_back(pairs[i].first);
    }

    // resa = [stracksa[i] for i in range(len(stracksa)) if i not in dupa]
    for (int i = 0; i < stracksa.size(); i++)
    {
        std::vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end())
        {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++)
    {
        std::vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end())
        {
            resb.push_back(stracksb[i]);
        }
    }
}

template<typename DetectionResult>
void BYTETracker<DetectionResult>::linear_assignment(std::vector< std::vector<float> >& cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
    std::vector< std::vector<int> >& matches, std::vector<int>& unmatched_a, std::vector<int>& unmatched_b)
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            unmatched_b.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol;  std::vector<int> colsol;
    float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);  // rowsol是跟踪器匹配的（检测框）坐标，colsol是检测框匹配的（跟踪器）坐标
    for (int i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            unmatched_b.push_back(i);
        }
    }
}

template<typename DetectionResult>
std::vector< std::vector<float> > BYTETracker<DetectionResult>::ious(std::vector< std::vector<float> >& atlbrs, std::vector< std::vector<float> >& btlbrs)
{
    std::vector< std::vector<float> > ious;
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    ious.resize(atlbrs.size());
    for (int i = 0; i < ious.size(); i++)
    {
        ious[i].resize(btlbrs.size());
    }

    //bbox_ious
    for (int k = 0; k < btlbrs.size(); k++)
    {
        std::vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
        for (int n = 0; n < atlbrs.size(); n++)
        {
            float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0)
            {
                float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if (ih > 0)
                {
                    float ua = (atlbrs[n][2] - atlbrs[n][0] + 1) * (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                }
                else
                {
                    ious[n][k] = 0.0;
                }
            }
            else
            {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

template<typename DetectionResult>
std::vector< std::vector<float> > BYTETracker<DetectionResult>::iou_distance(std::vector<STrack<DetectionResult>>& atracks, std::vector<STrack<DetectionResult>>& btracks)
{
    std::vector< std::vector<float> > atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i].tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    std::vector< std::vector<float> > _ious = ious(atlbrs, btlbrs);
    std::vector< std::vector<float> > cost_matrix;
    for (int i = 0; i < _ious.size(); i++)
    {
        std::vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

template<typename DetectionResult>
std::vector< std::vector<float> > BYTETracker<DetectionResult>::iou_distance(std::vector<STrack<DetectionResult>*>& atracks, std::vector<STrack<DetectionResult>>& btracks)
{
    std::vector< std::vector<float> > atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i]->tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    std::vector< std::vector<float> > _ious = ious(atlbrs, btlbrs);
    std::vector< std::vector<float> > cost_matrix;
    for (int i = 0; i < _ious.size(); i++)
    {
        std::vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

template<typename DetectionResult>
double BYTETracker<DetectionResult>::lapjv(const  std::vector< std::vector<float> >& cost, std::vector<int>& rowsol, std::vector<int>& colsol,
    bool extend_cost, float cost_limit, bool return_cost)
{
    std::vector< std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector< std::vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            std::cout << "set extend_cost=True" << std::endl;
            system("pause");
            exit(0);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double** cost_ptr;
    cost_ptr = new double* [sizeof(double*) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int* x_c = new int[sizeof(int) * n];
    int* y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        std::cout << "Calculate Wrong!" << std::endl;
        system("pause");
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        if (return_cost)
        {
            for (int i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    //cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    }
    else if (return_cost)
    {
        for (int i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}