#include <algorithm>
#include "FaceDetector.h"

bool FaceDetector::hasGPU = true;
bool FaceDetector::toUseGPU = true;
FaceDetector::FaceDetector():
        _nms(0.4),
        _threshold(0.6),
        _retinaface(false),
        Net(new ncnn::Net())
        //_mean_val{104.f, 117.f, 123.f},
{
}

inline void FaceDetector::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    
    }
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

FaceDetector::FaceDetector(const std::string &model_param, const std::string &model_bin, bool useGPU)
{
    _nms = 0.4;
    _threshold = 0.6;
    //_mean_val = {104.f, 117.f, 123.f};
    _retinaface = false;
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    Net = new ncnn::Net();
    Net->opt.use_vulkan_compute = toUseGPU;
    Net->opt.use_fp16_arithmetic = true;
    Init(model_param, model_bin);
}

void FaceDetector::Init(const std::string &model_param, const std::string &model_bin)
{
    
    Net->load_param(model_param.c_str());
    Net->load_model(model_bin.c_str());
}

void FaceDetector::Detect(cv::Mat& image, std::vector<bbox>& boxes)
{
    int MAXSIZE = 320;
    //float mean_val[3] = {104.f, 117.f, 123.f};
    float long_side = std::max(image.cols, image.rows);
    float scale = MAXSIZE / long_side;
    cv::Mat img_scale;
    cv::Mat img_padding(MAXSIZE, MAXSIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(image, img_scale, cv::Size(image.cols*scale, image.rows*scale));
    cv::Rect2f r(0, 0, img_scale.cols, img_scale.rows);
    img_scale.copyTo(img_padding(r));
    
    ncnn::Mat in = ncnn::Mat::from_pixels(img_padding.data, ncnn::Mat::PIXEL_BGR, img_padding.cols, img_padding.rows);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0}; 
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    
#if NCNN_VULKAN
    if (toUseGPU) {  // 消除提示
        ex.set_vulkan_compute(toUseGPU);
    }
#endif

    ex.input(0, in);
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract("output0", out);

    // class
    ex.extract("530", out1);

    //landmark
    ex.extract("529", out2);

    std::vector<box> anchor;
    if (_retinaface)
        create_anchor_retinaface(anchor, img_padding.cols, img_padding.rows);
    else
        create_anchor(anchor,  img_padding.cols, img_padding.rows);

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);
    
    if (ptr == nullptr || ptr1 == nullptr || landms == nullptr || anchor.empty()) {
    // Handle error here
    return;
    }

    // #pragma omp parallel for num_threads(2)
    for(size_t i = 0; i < anchor.size(); ++i){
        if ((ptr1+1) <= &ptr1[anchor.size()-1] && *(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.w;
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.h;
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    for(size_t j = 0; j < total_box.size(); ++j){
        bbox rescale_box;
        rescale_box.x1 = total_box[j].x1/scale;
        rescale_box.y1 = total_box[j].y1/scale;
        rescale_box.x2 = total_box[j].x2/scale;
        rescale_box.y2 = total_box[j].y2/scale;
        rescale_box.s = total_box[j].s;
        for (int k =0; k<5; k++)
        {
            rescale_box.point[k]._x = total_box[j].point[k]._x/scale;
            rescale_box.point[k]._y = total_box[j].point[k]._y/scale;
        }
        boxes.push_back(rescale_box);
    }
}

inline bool FaceDetector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void FaceDetector::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;

}

FaceDetector::~FaceDetector(){
    Release();
}

void FaceDetector::create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for(size_t i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for(size_t k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for(size_t l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void FaceDetector::create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for(size_t i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for(size_t k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for(size_t l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void FaceDetector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for(size_t i = 0; i < input_boxes.size(); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (size_t i = 0; i < input_boxes.size(); ++i)
    {
        for (size_t j = i + 1; j < input_boxes.size();)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
