

#ifndef TEST_NCNN2_COMMON_H
#define TEST_NCNN2_COMMON_H


#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <iostream>


#define MAX_STRIDE 64
#define PERMUTE 0

struct Object {
    cv::Rect_<float> rect;
    int label{};
    float prob{};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
    std::vector<cv::Point3f> key_points;
};

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

extern const unsigned char colors[81][3];

void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color);

void draw_pose(cv::Mat& bgr, std::vector<cv::Point3f> key_points);

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true);


#endif //TEST_NCNN2_COMMON_H
