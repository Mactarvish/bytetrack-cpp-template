# bytetrack-cpp-template
不同的业务中的检测实例往往会使用不同的数据结构来表示，因此需要一个通用的跟踪算法，能够适应不同的数据结构。这里我们使用模板化的方式来实现一个通用的跟踪算法，这样就可以在不同的业务中使用不同的数据结构来表示检测实例，而不需要修改跟踪算法的代码。


<div align="center">
  <img src="docs/readme/hand.gif"/>
</div>


## 使用方法
依赖：

编译算法库仅依赖Eigen3，运行demo需要opencv和jsoncpp

使用步骤：
1. 自定义检测实例的数据结构	

2. 实例化一个跟踪器

3. 在pipeline中调用跟踪器的update方法进行跟踪


自定义的检测实例需要满足以下条件：
1. 存在默认构造函数；

2. 存在以下签名的构造函数：
`DetectionResult(const int& tlx, const int& tly, const int& w, const int& h, int _cat, float _conf, int _trackId)`

其中参数依次为：检测框左上顶点X坐标，检测框左上顶点Y坐标，检测框宽度，检测框高度，类别，置信度，跟踪ID；

3. 存在公有方法
`std::tuple<int, int, int, int> GetXYWH() const`
，返回：检测框左上顶点X坐标，检测框左上顶点Y坐标，检测框宽度，检测框高度；

4. 存在公有成员变量`confidence`，`category`，`trackId`。

demo见bytetrack_demo
```
struct HandDetectionResult 
{
    HandDetectionResult() : x1(0), y1(0), x2(0), y2(0), x3(0), y3(0), x4(0), y4(0), confidence(0), category(-1), trackId(-1) {}
    
    HandDetectionResult(const int& tlx, const int& tly, const int& w, const int& h, int _cat, float _conf, int _trackId)
    {
		...
	}
    
    std::tuple<int, int, int, int> GetXYWH() const
    {
      ...
    }

    float confidence;
    int category;
    int trackId;

    ...
};



int main(int argc, char* argv[])
{
    ...
    // 构造跟踪器
    BYTETracker<HandDetectionResult> byteTracker(0.5f, 0.8f, 0.9f, 5, 3);

	for (const auto& srcImagePath : srcImagePaths)
	{
		// 执行检测，得到检测结果
		auto detectionResults = ...;
        // 执行跟踪
		auto trackingResults = byteTracker.update(detectionResults);
	}
}

```

