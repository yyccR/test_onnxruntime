
#include <iostream>
#include "memory"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

#include "../common/common.h"


//using YoloNasAnchor = std::vector<int>;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };


void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}


void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}


void generate_anchors(const int target_height, const int target_width, std::vector<int> &strides, std::vector<GridAndStride> &anchors)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_width / stride;
        int num_grid_h = target_height / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1)
        {
            for (int g0 = 0; g0 < num_grid_w; ++g0)
            {
                anchors.push_back((GridAndStride) {g0, g1, stride});
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const float* box_pred, const float* cls_pred, float prob_threshold, std::vector<BoxInfo>& boxes)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = cls_pred+i*num_class;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = *(scores+k);
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        if (score >= prob_threshold)
        {

//            float pred_ltrb[4];
//            const float *dis_after_sm = box_pred + i * 4;
//            pred_ltrb[0] = (-dis_after_sm[0] + grid_strides[i].grid0) * grid_strides[i].stride;
//            pred_ltrb[1] = (-dis_after_sm[1] + grid_strides[i].grid1) * grid_strides[i].stride;
//            pred_ltrb[2] = (dis_after_sm[2] + grid_strides[i].grid0) * grid_strides[i].stride;
//            pred_ltrb[3] = (dis_after_sm[3] + grid_strides[i].grid1) * grid_strides[i].stride;
//
//
//
//            Object obj;
//            obj.rect.x = (pred_ltrb[2] - pred_ltrb[0]) / 2;
//            obj.rect.y = (pred_ltrb[3] - pred_ltrb[1]) / 2;
//            obj.rect.width = pred_ltrb[2] - pred_ltrb[0];
//            obj.rect.height = pred_ltrb[3] - pred_ltrb[1];
//            obj.label = label;
//            obj.prob = score;
//            obj.mask_feat.resize(32);
//            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
//            objects.push_back(obj);

            BoxInfo box;
            const float *dis_after_sm = box_pred + i * 4;
//            box.x1 = (-dis_after_sm[0] + grid_strides[i].grid0) * grid_strides[i].stride;
//            box.y1 = (-dis_after_sm[1] + grid_strides[i].grid1) * grid_strides[i].stride;
//            box.x2 = (dis_after_sm[2] + grid_strides[i].grid0) * grid_strides[i].stride;
//            box.y2 = (dis_after_sm[3] + grid_strides[i].grid1) * grid_strides[i].stride;

            box.x1 = dis_after_sm[0];
            box.y1 = dis_after_sm[1];
            box.x2 = dis_after_sm[2];
            box.y2 = dis_after_sm[3];

            box.label = label;
            box.score = score;
//            obj.mask_feat.resize(32);
//            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            boxes.push_back(box);
        }
    }
}

int test_yolo_nas() {

    int input_shape = 640;
    std::string image_file("/Users/yang/CLionProjects/test_onnxruntime/data/images/traffic_road.jpg");
    cv::Mat image = cv::imread(image_file);
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(input_shape,input_shape));
    input_image.convertTo(input_image, CV_32F, 1.0/255.0);
    float w_scale = (float) image.cols / (float)input_shape;
    float h_scale = (float) image.rows / (float)input_shape;
    std::cout << "w_scale h_scale: " << w_scale << " " << h_scale << std::endl;

    std::string model_file("/Users/yang/CLionProjects/test_onnxruntime/yolo-nas/yolo_nas_s.onnx");
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test_mac");
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::unique_ptr<Ort::Session> session;
    session = std::make_unique<Ort::Session>(Ort::Session(env, model_file.c_str(), session_options));
//    Ort::Session session(env, model_file.c_str(), session_options);

    // 输入输出计数
    size_t input_size = session->GetInputCount();
    size_t output_size = session->GetOutputCount();

    // 定义输入输出名字,纬度变量
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;

    // 拿到输入输出名字和纬度,赋值给上述变量, onnxruntime 1.13 把GetInputName替换成了GetInputNameAllocated, 避免了内存泄漏问题
    Ort::AllocatorWithDefaultOptions allocator;
    for(size_t i = 0; i < input_size; i++){
        input_names.push_back(session->GetInputNameAllocated(i, allocator).release());
        auto input_type_info = session->GetInputTypeInfo(i);
        input_node_dims.push_back(input_type_info.GetTensorTypeAndShapeInfo().GetShape());
        std::cout << "input name: " << input_names[i] << std::endl;
        for(auto& s: input_type_info.GetTensorTypeAndShapeInfo().GetShape()){
            std::cout << "input shape: " << s << std::endl;
        }

    }
    for(size_t i = 0; i < output_size; i++){
        output_names.push_back(session->GetOutputNameAllocated(i, allocator).release());
        auto output_type_info = session->GetOutputTypeInfo(i);
        output_node_dims.push_back(output_type_info.GetTensorTypeAndShapeInfo().GetShape());
        std::cout << "output name: " << output_names[i] << std::endl;
        for(auto& s: output_type_info.GetTensorTypeAndShapeInfo().GetShape()){
            std::cout << "output shape: " << s << std::endl;
        }
    }

    for(auto& n : input_names){std::cout << n << std::endl;}
    for(auto& n : output_names){std::cout << n << std::endl;}

    // 输入数据
    // cv::mat格式转vector,再转Ort::Value, yolov5-lite只有一个输入，所以只处理第一个
    cv::Mat image_channels[3];
    cv::split(input_image, image_channels);
    std::vector<float> input_values;
    // BGR2RGB, HWC->CHW
    for (int i = 0; i < input_image.channels(); i++)
    {
        std::vector<float> data = std::vector<float>(image_channels[2 - i].reshape(1, input_image.cols * input_image.rows));
        input_values.insert(input_values.end(), data.begin(), data.end());
    }
    // Ort::Value, 只处理第一个input, yolo-nas只有一个输入
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_values.data(), input_values.size(),input_node_dims[0].data(), input_node_dims[0].size());


    // 推理
    std::vector<Ort::Value> ort_outputs = session->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor, 1, output_names.data(), output_names.size());
    // 拿到推理结果
    const float* box_preds = nullptr;
    const float* cls_preds = nullptr;
    for(int i=0; i < ort_outputs.size(); i++){
        if(ort_outputs[i].GetTensorTypeAndShapeInfo().GetShape()[2] == 4){
            box_preds = ort_outputs[i].GetTensorMutableData<float>();
        }else{
            cls_preds = ort_outputs[i].GetTensorMutableData<float>();
        }
    }
    const float* preds = ort_outputs[0].GetTensorMutableData<float>();

    int num_classes = 80;
    float conf_thres = 0.25;
    float nms_thres = 0.5;
    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_anchors(input_shape, input_shape, strides, grid_strides);

    std::vector<BoxInfo> boxes;
    generate_proposals(grid_strides, box_preds, cls_preds, conf_thres, boxes);

//    const int nout = num_classes + 5;

//    const float anchors[3][6] = { {10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
//    const float stride[3] = { 8.0, 16.0, 32.0 };


//    for (int n = 0; n < 3; n++)
//    {
//        int num_grid_x = (int)(input_shape / stride[n]);
//        int num_grid_y = (int)(input_shape / stride[n]);
//        for (int q = 0; q < 3; q++)
//        {
//            const float anchor_w = anchors[n][q * 2];
//            const float anchor_h = anchors[n][q * 2 + 1];
//            for (int i = 0; i < num_grid_y; i++)
//            {
//                for (int j = 0; j < num_grid_x; j++)
//                {
//                    float box_score = preds[4];
//                    if (box_score > conf_thres)
//                    {
//                        float class_score = 0;
//                        int class_ind = 0;
//                        for (int k = 0; k < num_classes; k++)
//                        {
//                            if (preds[k + 5] > class_score)
//                            {
//                                class_score = preds[k + 5];
//                                class_ind = k;
//                            }
//                        }
//                        //if (class_score > this->confThreshold)
//                        //{
//                        float cx = (preds[0] * 2.f - 0.5f + j) * stride[n];  ///cx
//                        float cy = (preds[1] * 2.f - 0.5f + i) * stride[n];   ///cy
//                        float w = powf(preds[2] * 2.f, 2.f) * anchor_w;   ///w
//                        float h = powf(preds[3] * 2.f, 2.f) * anchor_h;  ///h
//
//                        float xmin = std::max(0.0,(cx - 0.5 * w)*w_scale);
//                        float ymin = std::max(0.0,(cy - 0.5 * h)*h_scale);
//                        float xmax = std::min((float)image.cols, (float)(cx + 0.5 * w)*w_scale);
//                        float ymax = std::min((float)image.rows, (float)(cy + 0.5 * h)*h_scale);
//
//                        boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_score, class_ind });
//                        //}
//                    }
//                    preds += nout;
//                }
//            }
//        }
//    }

    for(BoxInfo& box: boxes){
        box.x1 = std::max(0.f,(float)(box.x1*w_scale));
        box.y1 = std::max(0.f,(float)(box.y1*h_scale));
        box.x2 = std::min((float)image.cols, (float)box.x2*w_scale);
        box.y2 = std::min((float)image.rows, (float)box.y2*h_scale);
    }

    nms(boxes,nms_thres);
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);


    return 0;
};
