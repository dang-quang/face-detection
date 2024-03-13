//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "ncnn/ncnn/net.h"
#include "utils.h"

class FaceDetector
{

public:
    FaceDetector();

    void Init(const std::string &model_param, const std::string &model_bin);

    FaceDetector(const std::string &model_param, const std::string &model_bin, bool UseGPU);

    inline void Release();

    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

    ~FaceDetector();

private:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;

    ncnn::Net *Net;

    void create_anchor(std::vector<box> &anchor, int w, int h);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    //int MAXSIZE = 320;
public:
    static FaceDetector *detector;
    static bool hasGPU;
    static bool toUseGPU;
    static int MAXSIZE;
};
#endif //FACE_DETECTOR_H
