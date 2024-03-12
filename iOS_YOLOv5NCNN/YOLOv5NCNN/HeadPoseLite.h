#ifndef _HEAHPOSELITE_H_
#define _HEAHPOSELITE_H_

#include "ncnn/ncnn/net.h"
// #include "common.h"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <numeric>

#define ODIM   66
#define NEAR_0 1e-10

class HeadPoseLite
{
public:
	HeadPoseLite();
	HeadPoseLite(const std::string &model_param, const std::string &model_bin, bool useGPU);
	~HeadPoseLite();
	int Predict(const cv::Mat& image, bbox& bbox, PoseValue& pose_value, ScaleInfo& scale_info);

private:
	ncnn::Net* hopenet;

	bool initialized_;
	const cv::Size inputSize = cv::Size(48,48);
	cv::Mat PreProcess(const cv::Mat& original_img, bbox& bbox, ScaleInfo& scaleinfo);
	void PostProcess(float* yaw, float* pitch, float* roll, PoseValue& pose_value);
	void softmax(float* z, size_t el);
	double getAngle(float* prediction, size_t len);
	int idx_tensor[ODIM];
public:
    static HeadPoseLite *detector;
    static bool hasGPU;
    static bool toUseGPU;
};



#endif // !_HEAHPOSELITE_H_
