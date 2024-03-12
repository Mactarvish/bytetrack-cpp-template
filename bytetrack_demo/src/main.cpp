#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "json/json.h"

#include "BYTETracker.h"


using namespace cv;
using namespace std;


struct HandDetectionResult 
{
    HandDetectionResult() : x1(0), y1(0), x2(0), y2(0), x3(0), y3(0), x4(0), y4(0), confidence(0), category(-1), trackId(-1) {}
    
    HandDetectionResult(const int& tlx, const int& tly, const int& w, const int& h, int _cat, float _conf, int _trackId)
        : x1(tlx), y1(tly), x2(tlx + w), y2(tly), x3(tlx + w), y3(tly + h), x4(tlx), y4(tly + h), confidence(_conf), category(_cat), trackId(_trackId) {}
    
    std::tuple<int, int, int, int> GetXYWH() const
    {
        int xMin = min(min(x1, x2), min(x3, x4));
        int yMin = min(min(y1, y2), min(y3, y4));
        int xMax = max(max(x1, x2), max(x3, x4));
        int yMax = max(max(y1, y2), max(y3, y4));
        auto x = xMin;
        auto y = yMin;
        auto w = xMax - xMin;
        auto h = yMax - yMin;
        return std::make_tuple(x, y, w, h);
    }

    float confidence;
    int category;
    int trackId;

    int x1;
    int y1;
    int x2;
    int y2;
    int x3;
    int y3;
    int x4;
    int y4;

};


const vector<Scalar> COLORS = { {240, 212, 127}, {226, 169, 0}, {212, 127, 254}, {198, 85, 127}, {183, 42, 0},
                         {169, 0, 254}, {155, 42, 127}, {141, 85, 0}, {127, 254, 254}, {113, 212, 127},
                         {99, 169, 0}, {85, 127, 254}, {71, 85, 127}, {56, 42, 0}, {42, 0, 254},
                         {28, 42, 127}, {14, 85, 0}, {0, 254, 254}, {14, 212, 127}, {28, 169, 0},
                         {42, 127, 254}, {56, 85, 127}, {71, 42, 0}, {85, 0, 254}, {99, 42, 127} };


Mat VisualizeResult(const Mat& srcImage, const vector<HandDetectionResult>& hdrs)
{
    Mat visFrame;
    srcImage.copyTo(visFrame);
    const vector<pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
            {0, 5}, {5, 6}, {6, 7}, {7, 8},
            {0, 9}, {9, 10}, {10, 11}, {11, 12},
            {0, 13}, {13, 14}, {14, 15}, {15, 16},
            {0, 17}, {17, 18}, {18, 19}, {19, 20},
            {5, 9}, {9, 13}, {13, 17},
    };

    for (const auto& hdr : hdrs)
    {
        // 可视化手掌检测结果
        if (hdr.category != -1)
        {
            auto [x, y, w, h] = hdr.GetXYWH();
            Rect rect = Rect(x, y, w, h);
            float score = hdr.confidence;
            int color_ind = hdr.category % COLORS.size();
            Scalar color = COLORS[color_ind];
            rectangle(visFrame, rect, color, 2);
            char s_text[80];
            std::snprintf(s_text, sizeof(s_text), "%.2f %d", round(score * 1e3) / 1e3, hdr.trackId);
            string label = to_string(hdr.category) + " " + s_text;
            int baseLine = 0;
            Size textSize = getTextSize(label, FONT_HERSHEY_PLAIN, 0.7, 1, &baseLine);
            baseLine += 2;
            rectangle(visFrame, Rect(rect.x, rect.y - textSize.height, textSize.width + 1, textSize.height + 1), COLORS[hdr.trackId % COLORS.size()], 3);
            putText(visFrame, label, Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 0.7, {255, 255, 255}, 1);
        }
    }

    return visFrame;
}


inline bool ReadJson(const string& srcJsonPath, Json::Value& r)
{
    r.clear();
    ifstream inFile(srcJsonPath.data(), ios::binary);
    if (!inFile.good())
    {
        return false;
    }
    Json::Reader reader;
    return reader.parse(inFile, r);
}

inline void WriteJson(const Json::Value& r, const string& jsonSavePath)
{
    // 不知道什么原因，有的时候写入会失败，这里循环读取检查写入是否成功
    int c = 0;
    Json::Value m;
    do
    {
        ofstream ofFile(jsonSavePath.data(), std::ios::out);
        Json::FastWriter writer;
        std::string jsonstr = writer.write(r);
        ofFile << jsonstr;
        c++;
    } while (c < 10 && !ReadJson(jsonSavePath, m));
}


vector<HandDetectionResult> LoadLabelmeAnnotation(const std::string& srcLabelmeJsonPath)
{
	Json::Value jr;
	ReadJson(srcLabelmeJsonPath, jr);
	vector<HandDetectionResult> rs;
	for (auto& e : jr["shapes"])
	{
		auto p1 = e["points"][0];
		auto p2 = e["points"][1];
		auto x1 = p1[0].asInt();
		auto y1 = p1[1].asInt();
		auto x2 = p2[0].asInt();
		auto y2 = p2[1].asInt();
        rs.push_back(HandDetectionResult{ x1, y1, x2 - x1, y2 - y1, 0, 0.9, -1 });
	}

	return rs;
}


int main(int argc, char* argv[])
{
    const string keys =
        "{help h usage ? |      | print this message   }"
        "{@srcImageDir        |  | source image directory to infer}"
        "{sj saveJson   |    | save infer result json}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.printMessage();

    string srcImageDir = parser.get<string>("@srcImageDir");
    bool saveJson = parser.get<bool>("saveJson");


    // 构造跟踪器
    BYTETracker<HandDetectionResult> byteTracker(0.5f, 0.8f, 0.9f, 5, 3);

    vector<string> srcImagePaths;
    if (srcImageDir.substr(srcImageDir.size() - 4) == ".jpg" || srcImageDir.substr(srcImageDir.size() - 4) == ".yuv" || srcImageDir.substr(srcImageDir.size() - 4) == ".png")
    {
        srcImagePaths = { srcImageDir };
    }
    else
    {
#ifdef WIN32
        srcImageDir += "\\";
#else
        srcImageDir += "/";
#endif
        glob(srcImageDir + "*.jpg", srcImagePaths, true);
        sort(srcImagePaths.begin(), srcImagePaths.end());
    }

	for (const auto& srcImagePath : srcImagePaths)
	{
		auto srcImageName = std::filesystem::path(srcImagePath).filename().string();
		auto srcLabelmeJsonPath = srcImagePath.substr(0, srcImagePath.size() - 4) + ".json";
		
		auto detectionResults = std::vector<HandDetectionResult>{};
		// 如果存在labelme的json文件，则直接读取labelme的json文件（作为检测结果）
		if (std::filesystem::exists(srcLabelmeJsonPath))
		{
            // 生成一个0~1内的随机数，如果小于0.1那么忽略json，从而模拟漏检的情况
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);
            if (dis(gen) > 0.1)
            {
				detectionResults = LoadLabelmeAnnotation(srcLabelmeJsonPath);
            }
		}

		auto visSavePath = (std::filesystem::path("visualization") /
			std::filesystem::path(srcImageName.substr(0, srcImageName.size() - 4) + ".jpg")).string();
		auto jsonSavePath = (std::filesystem::path("visualization") /
			std::filesystem::path(srcImageName.substr(0, srcImageName.size() - 4) + ".json")).string();

		cv::Mat srcImage = imread(srcImagePath);

		auto startTime = chrono::high_resolution_clock::now();
        // 执行跟踪
		auto trackingResults = byteTracker.update(detectionResults);

		auto endTime = chrono::high_resolution_clock::now();
		auto ms = chrono::duration_cast<chrono::microseconds>(endTime - startTime)
			.count();
		Mat visImage;

		srcImage.copyTo(visImage);

		if (!std::filesystem::exists("visualization"))
		{
			std::filesystem::create_directory("visualization");
		}

		visImage = VisualizeResult(visImage, trackingResults);
		if (!std::filesystem::exists("visualization"))
		{
			std::filesystem::create_directory("visualization");
		}
		imwrite(visSavePath, visImage);
		if (saveJson)
		{
			//WriteJson(ConvertTrackingResultToJson(trackingResults), jsonSavePath);
		}
	}


    return 0;
}
