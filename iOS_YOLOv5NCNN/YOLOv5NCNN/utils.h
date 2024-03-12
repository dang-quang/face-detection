#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <vector>

using namespace std;

struct PointX{
    float _x;
    float _y;
};

struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    PointX point[5];
};

struct box {
    float cx;
    float cy;
    float sx;
    float sy;
};

struct PoseValue {
        float yaw;
        float pitch;
        float roll;
};

struct ScaleInfo {
	float x_min, x_max, y_min, y_max, box_heigh, box_width;
};

struct HeadPoseInfo {
	std::vector<float> yaw;
	std::vector<float> pitch;
	std::vector<float> roll;
};

const int num_landmarks = 106;

struct LandmarkResult{
    PointX landmarks[num_landmarks];
};

struct FaceLandmarkPoseResult
{
    bbox bbox;
    PoseValue pose;
    LandmarkResult lmk;
};

#endif // !_UTILS_H_
