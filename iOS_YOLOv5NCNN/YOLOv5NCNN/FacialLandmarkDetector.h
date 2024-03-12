#ifndef _FACIALLANDMARKDETECTOR_H_
#define _FACIALLANDMARKDETECTOR_H_


// #include "common.h"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "ncnn/ncnn/net.h"

class FacialLandmarkDetector
{
public:
	FacialLandmarkDetector();
	FacialLandmarkDetector(const std::string &model_param, const std::string &model_bin, bool useGPU);
	~FacialLandmarkDetector();
	int Predict(const cv::Mat& image,const bbox& bbox, LandmarkResult& lmk_results);

private:
	ncnn::Net* landmark_net;

	bool initialized_;
	const cv::Size inputSize_ = cv::Size(112, 112);
	const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
	
	cv::Mat PreProcess(const cv::Mat& original_img,const bbox& bbox, ScaleInfo& scaleinfo);
	void PostProcess(ncnn::Mat& out, const ScaleInfo& scale_info, LandmarkResult& lmk_results);
public:
    static FacialLandmarkDetector *detector;
    static bool hasGPU;
    static bool toUseGPU;	
};



#endif // !_FACIALLANDMARKDETECTOR_H_
