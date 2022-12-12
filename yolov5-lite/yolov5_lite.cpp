
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;


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



int main() {

    std::string image_file("/Users/yang/CLionProjects/test_onnxruntime/data/images/bus.jpeg");
    cv::Mat image = cv::imread(image_file);
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(416,416));
    float w_scale = (float) image.cols / 416.f;
    float h_scale = (float) image.rows / 416.f;

    std::string model_file("/Users/yang/CLionProjects/test_onnxruntime/yolov5-lite/v5Lite-s-sim-416.onnx");
    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "test_mac");
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_file.c_str(), session_options);

    // 创建输入数组
    std::vector<int64_t> input_tensor_dims = {1,3,416,416};  // 输入张量的维度
    const size_t num_input_nodes = session.GetInputCount();
    // 定义两个向量，用于存储输入节点的名称和维度信息
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    // 预留足够的空间来存储所有输入节点的信息
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);

    // 获取输入节点的名称
    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_names_ptr.emplace_back(session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        input_node_names.emplace_back(input_names_ptr[i].get());
    }
    for(auto &i: input_node_names){
        std::cout << std::string(i) << std::endl;
    }

    // 获取输入节点的维度信息
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    const auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_node_dims = tensor_info.GetShape();
    for(auto &i : input_node_dims){
        std::cout << i << std::endl;
    }

    // 定义内存分配器
    Ort::AllocatorWithDefaultOptions allocator;

    // 创建输入节点
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i) {
        // 将输入节点的信息包装成 Ort::Value
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> input_tensor_data(3*416*416);
        cv::Mat image_channels[3];
        cv::split(input_image, image_channels);
        for (int j = 0; j < 3; j++) {
            memcpy(input_tensor_data.data() + 416*416 * j, image_channels[j].data,416*416 * sizeof(float));
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_data.data(), 3*416*416,
                                                            input_node_dims.data(), 4);
        input_tensors.push_back(input_tensor);
    }

    // 创建输出节点
//    std::vector<Ort::Value> output_tensors;

    // 使用会话运行机器学习模型
//    Ort::RunOptions run_options;
//    session.Run(run_options, input_node_names.data(), input_tensors.data(), num_input_nodes, output_node_names.data(), output_node_names.size(), output_tensors.data());




    return 0;
};
