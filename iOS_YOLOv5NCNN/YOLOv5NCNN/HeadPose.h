#ifndef _HEADPOSE_H_
#define _HEADPOSE_H_

#include "ncnn/ncnn/net.h"
// #include "common.h"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "utils.h"

class HeadPose
{
public:
	HeadPose();
	HeadPose(const std::string &model_param, const std::string &model_bin, bool useGPU);
	~HeadPose();
	int Predict(const cv::Mat& image, bbox& bbox, PoseValue& pose_value, ScaleInfo& scale_info);

private:
	ncnn::Net* hopenet;

	bool initialized_;
	const cv::Size inputSize_ = { 224, 224 };
	const float mean[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f}; // image = (image - mean) / std
	const float norm[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f}; // convert back before transform: ((image * std) + mean)
	cv::Mat PreProcess(const cv::Mat& original_img, bbox& bbox, ScaleInfo& scaleinfo);
	void PostProcess(std::vector<float> yaw, std::vector<float> pitch, std::vector<float> roll, PoseValue& pose_value);
	void softmax(std::vector<float> &input, std::vector<float>& output);
public:
    static HeadPose *detector;
    static bool hasGPU;
    static bool toUseGPU;
};

#endif // !_HEAHPOSE_H_
