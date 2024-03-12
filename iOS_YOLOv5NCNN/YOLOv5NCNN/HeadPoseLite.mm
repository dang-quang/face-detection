#include "HeadPoseLite.h"
#include <iostream>
#include <math.h>

bool HeadPoseLite::hasGPU = true;
bool HeadPoseLite::toUseGPU = true;
// Constructor
HeadPoseLite::HeadPoseLite(){
    hopenet = new ncnn::Net();
    for (uint i=1; i<ODIM+1; i++) idx_tensor[i-1] = i;
}

HeadPoseLite::HeadPoseLite(const std::string &model_param, const std::string &model_bin, bool useGPU){
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    hopenet = new ncnn::Net();
    hopenet->opt.use_vulkan_compute = toUseGPU;
    hopenet->opt.use_fp16_arithmetic = true;
    hopenet->load_param(model_param.c_str());
    hopenet->load_model(model_bin.c_str());
    for (uint i=1; i<ODIM+1; i++) idx_tensor[i-1] = i;
}

// Deconstructor
HeadPoseLite::~HeadPoseLite() {
    if (hopenet) {
        hopenet->clear();
    }
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

cv::Mat HeadPoseLite::PreProcess(const cv::Mat& original_img, bbox& bbox, ScaleInfo& scaleinfo)
{
    float x_min = bbox.x1;
    float y_min = bbox.y1;
    float x_max = bbox.x2;
    float y_max = bbox.y2;
    // scaling bouding box
    float bbox_width = abs(x_max - x_min);
    float bbox_height = abs(y_max - y_min);
    //
    float width_scale = static_cast<float>(2 * bbox_width / 4);
    float height_scale = static_cast<float>(bbox_height / 4);
    //
    float x_min_new =  x_min - width_scale;
    float x_max_new =  x_max + width_scale;
    float y_min_new = y_min - 3 * height_scale;
    float y_max_new = y_max + height_scale;
    // Get bbox to crop image
    float min_threshold = 0.0, tl_x, tl_y, br_x, br_y;
    tl_x = std::max(x_min_new, min_threshold);
    tl_y = std::max(y_min_new, min_threshold);
    float x_max_t = std::min(static_cast<float>(original_img.cols), x_max_new);
    float y_max_t = std::min(static_cast<float>(original_img.rows), y_max_new);
    br_x = x_max_t - tl_x; // subtract margin coords to x, y
    br_y = y_max_t - tl_y;
    
    // Get bbox values
    scaleinfo.box_heigh = bbox_height;
    scaleinfo.box_width = bbox_width;
    scaleinfo.x_max=x_max_t;
    scaleinfo.y_max=y_max_t;
    scaleinfo.x_min=tl_x;
    scaleinfo.y_min=tl_y;
    //
    cv::Rect roi = cv::Rect(tl_x, tl_y, br_x, br_y);
    //
    // input_img = img_cpy(cv::Range(roi.y, roi.height), cv::Range(roi.x, roi.width));
    
    cv::Mat input_img = original_img(roi).clone();
    cv::Mat faceGray;
    cv::cvtColor(input_img, faceGray, 6);
    cv::Mat faceGrayResized;
    cv::resize(faceGray, faceGrayResized, inputSize);
    return faceGrayResized;
}
// Post processing network output
void HeadPoseLite::PostProcess(float* yaw, float* pitch,
                            float* roll, PoseValue& pose_value)
{
    softmax(pitch, ODIM);
    softmax(yaw,  ODIM);
    softmax(roll,   ODIM);
    pose_value.yaw = getAngle(yaw, ODIM);
    pose_value.pitch = getAngle(pitch, ODIM);
    pose_value.roll = getAngle(roll, ODIM);
}
// predict yaw, pitch, roll from image
int HeadPoseLite::Predict(const cv::Mat& image, bbox& bbox, PoseValue& pose_value, ScaleInfo& scale_info)
{
    cv::Mat img_cpy = image.clone();

    cv::Mat input_img = PreProcess(image, bbox, scale_info);

    ncnn::Extractor ex = hopenet->create_extractor();

    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(input_img.data,
    //     ncnn::Mat::PIXEL_BGR2RGB, input_img.cols, input_img.rows, inputSize_.width, inputSize_.height);
    // https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result#pre-process
    ncnn::Mat in = ncnn::Mat::from_pixels(input_img.data, ncnn::Mat::PIXEL_GRAY, 48, 48);
    // in.substract_mean_normalize(mean, norm);
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("data", in);
    //
    ncnn::Mat output;
    ex.extract("hybridsequential0_multitask0_dense0_fwd", output);
    float* pred_pitch = output.range(0, ODIM);
    float* pred_roll  = output.range(ODIM, ODIM*2);
    float* pred_yaw   = output.range(ODIM*2, ODIM*3);
    
    // std::string yaw_layer_name = "509";
    // std::string pitch_layer_name = "510";
    // std::string roll_layer_name = "511";

    // ncnn::Mat yaw_mat, pitch_mat, roll_mat;
    // ex.extract(yaw_layer_name.c_str(), yaw_mat);
    // ex.extract(pitch_layer_name.c_str(), pitch_mat);
    // ex.extract(roll_layer_name.c_str(), roll_mat);
    // // ncnn mat to array
    // std::vector<float> yaw, pitch, roll;
    // HeadInfo head_info;
    // for (int j = 0; j < 66; j++)
    // {
    //     yaw.push_back(yaw_mat[j]);
    //     pitch.push_back(pitch_mat[j]);
    //     roll.push_back(roll_mat[j]);
    // }
    // // Post process output state
    PostProcess(pred_yaw, pred_pitch, pred_roll, pose_value);

    return 0;
}

void HeadPoseLite::softmax(float* z, size_t el) {
 double zmax = -INFINITY;
 double zsum = 0;
 for (size_t i = 0; i < el; i++) if (z[i] > zmax) zmax=z[i];
 for (size_t i=0; i<el; i++) z[i] = std::exp(z[i]-zmax);
 zsum = std::accumulate(z, z+el, 0.0);
 for (size_t i=0; i<el; i++) z[i] = (z[i]/zsum)+NEAR_0;
}

double HeadPoseLite::getAngle(float* prediction, size_t len)
{
  double expectation[len];
  for (uint i=0; i<len; i++) expectation[i]=idx_tensor[i]*prediction[i];
  double angle = std::accumulate(expectation, expectation+len, 0.0) * 3 - 99;
  return angle;
}
// void HeadPoseLite::softmax(std::vector<float> &input, std::vector<float>& output) 
// {

//     int i;
//     float m, sum, constant;

//     m = -INFINITY;
//     for (i = 0; i < input.size(); ++i) {
//         if (m < input[i]) {
//             m = input[i];
//         }
//     }

//     sum = 0.0;
//     for (i = 0; i < input.size(); ++i) {
//         sum += exp(input[i] - m);
//     }

//     constant = m + log(sum);
//     for (i = 0; i < input.size(); ++i) {
//         output.push_back(exp(input[i] - constant));
//     }

// }

/*
    End of namespace
*/

