
#include <iostream>
#include "memory"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

using Yolov5LiteAnchor = std::vector<int>;

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


void generate_anchors(const int target_height, const int target_width, std::vector<int> &strides, std::vector<Yolov5LiteAnchor> &anchors)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_width / stride;
        int num_grid_h = target_height / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1)
        {
            for (int g0 = 0; g0 < num_grid_w; ++g0)
            {
                anchors.push_back((Yolov5LiteAnchor) {g0, g1, stride});
            }
        }
    }
}

int main() {

    std::string image_file("/Users/yang/CLionProjects/test_onnxruntime/data/images/bus.jpeg");
    cv::Mat image = cv::imread(image_file);
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(416,416));
    float w_scale = (float) image.cols / 416.f;
    float h_scale = (float) image.rows / 416.f;

    std::string model_file("/Users/yang/CLionProjects/test_onnxruntime/yolov5-lite/v5Lite-s-sim-416.onnx");
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test_mac");
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_file.c_str(), session_options);

    // 输入输出计数
    size_t input_size = session.GetInputCount();
    size_t output_size = session.GetOutputCount();

    // 定义输入输出名字,纬度变量
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;

    // 拿到输入输出名字和纬度,赋值给上述变量, onnxruntime 1.13 把GetInputName替换成了GetInputNameAllocated, 避免了内存泄漏问题
    Ort::AllocatorWithDefaultOptions allocator;
    for(size_t i = 0; i < input_size; i++){
        input_names.push_back(session.GetInputNameAllocated(i, allocator).release());
        auto input_type_info = session.GetInputTypeInfo(i);
        input_node_dims.push_back(input_type_info.GetTensorTypeAndShapeInfo().GetShape());
    }
    for(size_t i = 0; i < output_size; i++){
        output_names.push_back(session.GetOutputNameAllocated(i, allocator).release());
        auto output_type_info = session.GetOutputTypeInfo(i);
        output_node_dims.push_back(output_type_info.GetTensorTypeAndShapeInfo().GetShape());
    }

    // cv::mat格式转vector,再转Ort::Value, yolov5-lite只有一个输入，所以只处理第一个
    input_image.convertTo(input_image, CV_32F, 1.0 / 255);  //divided by 255
    cv::Mat image_channels[3];
    cv::split(input_image, image_channels);
    std::vector<float> input_values;
    // BGR2RGB, HWC->CHW
    for (int i = 0; i < input_image.channels(); i++)
    {
        std::vector<float> data = std::vector<float>(image_channels[2 - i].reshape(1, input_image.cols * input_image.rows));
        input_values.insert(input_values.end(), data.begin(), data.end());
    }
    // Ort::Value, 只处理第一个input, yolov5-lite只有一个输入
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_values.data(), input_values.size(),input_node_dims[0].data(), input_node_dims[0].size());

    // 执行推理
    for(auto& n : input_names){
        std::cout << n << std::endl;
    }
    for(auto& n: input_node_dims){
        for(auto& i : n){
            std::cout << i << std::endl;
        }
    }
    for(auto& n : output_names){
        std::cout << n << std::endl;
    }
    for(auto& n: output_node_dims){
        for(auto& i : n){
            std::cout << i << std::endl;
        }
    }
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor, 1, output_names.data(), output_names.size());
    // 拿到推理结果
    const float* preds = ort_outputs[0].GetTensorMutableData<float>();

    std::vector<BoxInfo> generate_boxes;
    int n = 0, q = 0, i = 0, j = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
    int num_class = 80;
    const int nout = num_class + 5;
    const float anchors[3][6] = { {10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
    const float stride[3] = { 8.0, 16.0, 32.0 };
    float objThreshold = 0.6;
    for (n = 0; n < 3; n++)   ///����ͼ�߶�
    {
        int num_grid_x = (int)(416 / stride[n]);
        int num_grid_y = (int)(416 / stride[n]);
        for (q = 0; q < 3; q++)    ///anchor
        {
            const float anchor_w = anchors[n][q * 2];
            const float anchor_h = anchors[n][q * 2 + 1];
            for (i = 0; i < num_grid_y; i++)
            {
                for (j = 0; j < num_grid_x; j++)
                {
                    float box_score = preds[4];
                    if (box_score > objThreshold)
                    {
                        float class_score = 0;
                        int class_ind = 0;
                        for (k = 0; k < num_class; k++)
                        {
                            if (preds[k + 5] > class_score)
                            {
                                class_score = preds[k + 5];
                                class_ind = k;
                            }
                        }
                        //if (class_score > this->confThreshold)
                        //{
                        float cx = (preds[0] * 2.f - 0.5f + j) * stride[n];  ///cx
                        float cy = (preds[1] * 2.f - 0.5f + i) * stride[n];   ///cy
                        float w = powf(preds[2] * 2.f, 2.f) * anchor_w;   ///w
                        float h = powf(preds[3] * 2.f, 2.f) * anchor_h;  ///h

                        float xmin = (cx - 0.5 * w)*w_scale;
                        float ymin = (cy - 0.5 * h)*h_scale;
                        float xmax = (cx + 0.5 * w)*w_scale;
                        float ymax = (cy + 0.5 * h)*h_scale;

                        generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_score, class_ind });
                        //}
                    }
                    preds += nout;
                }
            }
        }
    }

    nms(generate_boxes,0.2);
    draw_coco_bboxes(image, generate_boxes);
    cv::waitKey(0);



    return 0;
};
