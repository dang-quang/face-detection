#include "FacialLandmarkDetector.h"
#include <iostream>
#include <math.h>


bool FacialLandmarkDetector::hasGPU = true;
bool FacialLandmarkDetector::toUseGPU = true;
// Constructor
FacialLandmarkDetector::FacialLandmarkDetector(){
    landmark_net = new ncnn::Net();
}

FacialLandmarkDetector::FacialLandmarkDetector(const std::string &model_param, const std::string &model_bin, bool useGPU){
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    landmark_net = new ncnn::Net();
    landmark_net->opt.use_vulkan_compute = toUseGPU;
    landmark_net->opt.use_fp16_arithmetic = true;
    landmark_net->load_param(model_param.c_str());
    landmark_net->load_model(model_bin.c_str());
}

// Deconstructor
FacialLandmarkDetector::~FacialLandmarkDetector() {
    if (landmark_net) {
        landmark_net->clear();
    }
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

cv::Mat FacialLandmarkDetector::PreProcess(const cv::Mat& original_img, const bbox& bbox, ScaleInfo& scaleinfo)
{
    int capture_width = original_img.cols-1;
    int capture_height = original_img.rows-1;
    float x = bbox.x1;
    float y = bbox.y1;
    float w = bbox.x2 - bbox.x1;
    float h = bbox.y2 - bbox.y1;
    float shift = w * 0.1;
    x = std::max(0.0f, x - shift);
    y = std::max(0.0f, y - shift);
    w = w + shift * 2;
    h = h + shift * 2;

    w = std::min(w,(float)(capture_width-x));
    h = std::min(h,(float)(capture_height-y));
    //
    cv::Rect roi = cv::Rect(x, y, w, h);//br_x, br_y);
    scaleinfo.x_min = x;
    scaleinfo.y_min = y;
    scaleinfo.box_width = w;
    scaleinfo.box_heigh = h;
    //
    // input_img = img_cpy(cv::Range(roi.y, roi.height), cv::Range(roi.x, roi.width));
    
    cv::Mat input_img = original_img(roi).clone();
    return input_img;
}
// Post processing network output
void FacialLandmarkDetector::PostProcess(ncnn::Mat& out, const ScaleInfo& scale_info, LandmarkResult& lmk_results)
{
    // int w = image.cols;
    // int h = image.rows;
    for (int i = 0; i < num_landmarks; i++)
    {
        lmk_results.landmarks[i]._x = out[2*i] * scale_info.box_width + scale_info.x_min;
        lmk_results.landmarks[i]._y = out[2*i+1] * scale_info.box_heigh + scale_info.y_min;
    }
}
// predict yaw, pitch, roll from image
int FacialLandmarkDetector::Predict(const cv::Mat& image, const bbox& bbox, LandmarkResult& lmk_results)
{
    cv::Mat img_cpy = image.clone();
    ScaleInfo scale_info;

    cv::Mat input_img = PreProcess(img_cpy, bbox, scale_info);
    cv::resize(input_img, input_img, inputSize_);
    ncnn::Mat out;
    ncnn::Mat in = ncnn::Mat::from_pixels(input_img.data, ncnn::Mat::PIXEL_BGR2RGB, inputSize_.height, inputSize_.width);

    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = landmark_net->create_extractor();

    ex.set_num_threads(8);
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("input_1", in);
    ex.extract("415", out);
    PostProcess(out, scale_info, lmk_results);
    

    return 0;
}



/*
    End of namespace
*/

