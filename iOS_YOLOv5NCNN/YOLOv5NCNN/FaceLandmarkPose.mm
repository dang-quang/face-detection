#include "FaceLandmarkPose.h"
#include "FaceDetector.h"

std::string FaceLandmarkPose::convertNSStringToStdString(NSString* nsString)
{
    if (!nsString) {
        return std::string();
    }
    
    const char* utf8String = [nsString UTF8String];
    if (!utf8String) {
        return std::string();
    }
    
    return std::string(utf8String);
}

bool FaceLandmarkPose::hasGPU = true;
bool FaceLandmarkPose::toUseGPU = true;
FaceLandmarkPose *FaceLandmarkPose::detector = nullptr;

FaceLandmarkPose::FaceLandmarkPose(bool useGPU) {
    mFaceDetector = new FaceDetector();
    mHeadPose = new HeadPose();
    mFacialLandmarkDetector = new FacialLandmarkDetector();
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    FaceNet = new ncnn::Net();
    FaceNet->opt.use_vulkan_compute = toUseGPU;
    FaceNet->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"face" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"face" ofType:@"bin"];

    int rp = FaceNet->load_param([parmaPath UTF8String]);
    int rm = FaceNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model face net success!\n");
    } else {
        fprintf(stderr, "net load fail face net,param:%d model:%d", rp, rm);
    }
    
    RobustAlphaNet = new ncnn::Net();
    RobustAlphaNet->opt.use_vulkan_compute = toUseGPU;
    RobustAlphaNet->opt.use_fp16_arithmetic = true;
    parmaPath = [[NSBundle mainBundle] pathForResource:@"robust_alpha_opt" ofType:@"param"];
    binPath = [[NSBundle mainBundle] pathForResource:@"robust_alpha_opt" ofType:@"bin"];
    rp = RobustAlphaNet->load_param([parmaPath UTF8String]);
    rm = RobustAlphaNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model robust alpha net success!\n");
    } else {
        fprintf(stderr, "net load fail robust alpha net,param:%d model:%d", rp, rm);
    }

    PFLDSimNet = new ncnn::Net();
    PFLDSimNet->opt.use_vulkan_compute = toUseGPU;
    PFLDSimNet->opt.use_fp16_arithmetic = true;
    parmaPath = [[NSBundle mainBundle] pathForResource:@"pfld-sim" ofType:@"param"];
    binPath = [[NSBundle mainBundle] pathForResource:@"pfld-sim" ofType:@"bin"];
    rp = PFLDSimNet->load_param([parmaPath UTF8String]);
    rm = PFLDSimNet->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model pfld sim net success!\n");
    } else {
        fprintf(stderr, "net load fail pfld sim net,param:%d model:%d", rp, rm);
    }
}

std::vector<FaceLandmarkPoseResult> FaceLandmarkPose::detect(UIImage *image){
    std::vector<FaceLandmarkPoseResult> final_results;

    int img_w = image.size.width;
    int img_h = image.size.height;
    int target_w = 300;
    int target_h = 300;
    unsigned char* rgba = new unsigned char[img_w * img_h * 4];
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGContextRef contextRef = CGBitmapContextCreate(rgba, img_w, img_h, 8, img_w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, img_w, img_h), image.CGImage);
    CGContextRelease(contextRef);
    ncnn::Mat src_img = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, img_w, img_h, target_w, target_h);
    cv::Mat cv_image(src_img.h, src_img.w, CV_8UC3);
    src_img.to_pixels(cv_image.data, ncnn::Mat::PIXEL_RGB2BGR);
    
    std::vector<bbox> face_results;
    mFaceDetector->Detect(cv_image,face_results);
    delete[] rgba;
    for (int i = 0; i < face_results.size(); i++)
    {
        PoseValue pose_data;
        ScaleInfo scale_data;
        bbox box_data;
        LandmarkResult lmk_data;
        mHeadPose->Predict(cv_image,face_results[i],pose_data,scale_data);
        mFacialLandmarkDetector->Predict(cv_image,face_results[i],lmk_data);
        //drawResults(cv_image, box_data, pose_data, scale_data, lmk_data);
        
        FaceLandmarkPoseResult tmp;
        tmp.bbox = face_results[i];
        tmp.pose = pose_data;
        tmp.lmk = lmk_data;
        final_results.push_back(tmp);
    }

    return final_results;
}

FaceLandmarkPose::~FaceLandmarkPose()
{
    delete mFaceDetector;
    delete mHeadPose;
    delete mFacialLandmarkDetector;
}

void FaceLandmarkPose::draw_axis(cv::Mat& img, float yaw, float pitch, float roll, int tdx, int tdy, int size)
{
    float pitch_n = pitch * M_PI / 180;
    float yaw_n = -(yaw * M_PI / 180);
    float roll_n = roll * M_PI / 180;

    // X-Axis pointing to right. drawn in red
    float x1 = size * (cos(yaw_n) * cos(roll_n)) + tdx;
    float y1 = size * (cos(pitch_n) * sin(roll_n) + cos(roll_n) * sin(pitch_n) * sin(yaw_n)) + tdy;

    // Y-Axis | drawn in green
    //        v
    float x2 = size * (-cos(yaw_n) * sin(roll_n)) + tdx;
    float y2 = size * (cos(pitch_n) * cos(roll_n) - sin(pitch_n) * sin(yaw_n) * sin(roll_n)) + tdy;

    // Z-Axis (out of the screen) drawn in blue
    float x3 = size * (sin(yaw_n)) + tdx;
    float y3 = size * (-cos(yaw_n) * sin(pitch_n)) + tdy;
    //
    cv::Point origin_point(tdx, tdy);
    cv::Point red(static_cast<int>(x1), static_cast<int>(y1));
    cv::Point green(static_cast<int>(x2), static_cast<int>(y2));
    cv::Point blue(static_cast<int>(x3), static_cast<int>(y3));
    // opencv c++ : RGB - python: BGR
    cv::line(img, origin_point, red, CV_RGB(0, 0, 255), 3);
    cv::line(img, origin_point, green, CV_RGB(0, 255, 0), 3);
    cv::line(img, origin_point, blue, CV_RGB(255, 0, 0), 2); 
}

void FaceLandmarkPose::drawResults(cv::Mat& image, bbox& boxes, PoseValue& pose_data, ScaleInfo& scale_data, LandmarkResult& lmk_data)
{
    int tdx_ = static_cast<int>((scale_data.x_min + scale_data.x_max) / 2);
    int tdy_ = static_cast<int>((scale_data.y_min + scale_data.y_max) / 2);
    int size_ = static_cast<int>(scale_data.box_heigh/2);
    float v_yaw = pose_data.yaw, v_pitch = pose_data.pitch, v_roll = pose_data.roll;
    draw_axis(image, v_yaw, v_pitch, v_roll, tdx_, tdy_, size_);

    cv::putText(image, "yaw: "+std::to_string(v_yaw).substr(0, std::to_string(v_yaw).find(".")+3), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 255), 2);
    cv::putText(image, "pitch: "+std::to_string(v_pitch).substr(0, std::to_string(v_pitch).find(".")+3), cv::Point(50, 80), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 2);
    cv::putText(image, "roll: "+std::to_string(v_roll).substr(0, std::to_string(v_roll).find(".")+3), cv::Point(50, 110), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);
    //cv::Rect draw_box = cv::Rect(boxes[i].x1, boxes[i].y1, boxes[i].x2 - boxes[i].x1, boxes[i].y2 - boxes[i].y1);
    cv::Rect draw_box = cv::Rect(boxes.x1, boxes.y1, boxes.x2 - boxes.x1, boxes.y2 - boxes.y1);
    cv::rectangle(image, draw_box, cv::Scalar(255,255,0),1,8);
    for(int i = 0; i < num_landmarks; i++){
        cv::Point point = cv::Point(lmk_data.landmarks[i]._x,lmk_data.landmarks[i]._y);
        cv::circle(image, point,2,cv::Scalar(0, 0, 255), -1);
    }
}
