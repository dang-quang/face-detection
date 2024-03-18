#ifndef FACE_LANDMARK_POSE_H
#define FACE_LANDMARK_POSE_H

#include <stdio.h>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "FaceDetector.h"
#include "HeadPose.h"
#include "HeadPoseLite.h"
#include "FacialLandmarkDetector.h"
#include "utils.h"
#include <string>
#include <opencv2/core/core.hpp> 

class FaceLandmarkPose {
public:
    FaceLandmarkPose(bool useGPU);
    ~FaceLandmarkPose();
    std::vector<FaceLandmarkPoseResult> detect(UIImage *image);
    UIImage *resizeImage(UIImage *image, CGFloat width, CGFloat height);

private:
    FaceDetector* mFaceDetector;
    HeadPose* mHeadPose;
    FacialLandmarkDetector* mFacialLandmarkDetector;
    ncnn::Net *FaceNet;
    ncnn::Net *RobustAlphaNet;
    ncnn::Net *PFLDSimNet;
    static std::string convertNSStringToStdString(NSString* nsString);
    void drawResults(cv::Mat& image, bbox& box,PoseValue& pose_data, ScaleInfo& scale_data, LandmarkResult& lmk_data);
    void draw_axis(cv::Mat& img, float yaw, float pitch, float roll, int tdx, int tdy, int size);
public:
    static FaceLandmarkPose *detector;
    static bool hasGPU;
    static bool toUseGPU;
};
#endif //FACE_LANDMARK_POSE_H
