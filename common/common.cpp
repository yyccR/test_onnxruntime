

#include "common.h"


const unsigned char colors[81][3] = {
        {56,  0,   255},
        {226, 255, 0},
        {0,   94,  255},
        {0,   37,  255},
        {0,   255, 94},
        {255, 226, 0},
        {0,   18,  255},
        {255, 151, 0},
        {170, 0,   255},
        {0,   255, 56},
        {255, 0,   75},
        {0,   75,  255},
        {0,   255, 169},
        {255, 0,   207},
        {75,  255, 0},
        {207, 0,   255},
        {37,  0,   255},
        {0,   207, 255},
        {94,  0,   255},
        {0,   255, 113},
        {255, 18,  0},
        {255, 0,   56},
        {18,  0,   255},
        {0,   255, 226},
        {170, 255, 0},
        {255, 0,   245},
        {151, 255, 0},
        {132, 255, 0},
        {75,  0,   255},
        {151, 0,   255},
        {0,   151, 255},
        {132, 0,   255},
        {0,   255, 245},
        {255, 132, 0},
        {226, 0,   255},
        {255, 37,  0},
        {207, 255, 0},
        {0,   255, 207},
        {94,  255, 0},
        {0,   226, 255},
        {56,  255, 0},
        {255, 94,  0},
        {255, 113, 0},
        {0,   132, 255},
        {255, 0,   132},
        {255, 170, 0},
        {255, 0,   188},
        {113, 255, 0},
        {245, 0,   255},
        {113, 0,   255},
        {255, 188, 0},
        {0,   113, 255},
        {255, 0,   0},
        {0,   56,  255},
        {255, 0,   113},
        {0,   255, 188},
        {255, 0,   94},
        {255, 0,   18},
        {18,  255, 0},
        {0,   255, 132},
        {0,   188, 255},
        {0,   245, 255},
        {0,   169, 255},
        {37,  255, 0},
        {255, 0,   151},
        {188, 0,   255},
        {0,   255, 37},
        {0,   255, 0},
        {255, 0,   170},
        {255, 0,   37},
        {255, 75,  0},
        {0,   0,   255},
        {255, 207, 0},
        {255, 0,   226},
        {255, 245, 0},
        {188, 255, 0},
        {0,   255, 18},
        {0,   255, 75},
        {0,   255, 151},
        {255, 56,  0},
        {245, 255, 0}
};

void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color) {
    for (int y = 0; y < bgr.rows; y++) {
        uchar* image_ptr = bgr.ptr(y);
        const float* mask_ptr = mask.ptr<float>(y);
        for (int x = 0; x < bgr.cols; x++) {
            if (mask_ptr[x] >= 0.5) {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
            }
            image_ptr += 3;
        }
    }
}

void draw_pose(cv::Mat& bgr, std::vector<cv::Point3f> key_points){

    cv::Point neck((key_points[5].x + key_points[6].x)/2.0, (key_points[5].y + key_points[6].y)/2.0);

    if(key_points[0].z > 0.5 && key_points[1].z > 0.5){
        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), cv::Point(key_points[1].x, key_points[1].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[0].z > 0.5 && key_points[2].z > 0.5){
        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), cv::Point(key_points[2].x, key_points[2].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[1].z > 0.5 && key_points[3].z > 0.5){
        cv::line(bgr, cv::Point(key_points[1].x, key_points[1].y), cv::Point(key_points[3].x, key_points[3].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[2].z > 0.5 && key_points[4].z > 0.5){
        cv::line(bgr, cv::Point(key_points[2].x, key_points[2].y), cv::Point(key_points[4].x, key_points[4].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[2].z > 0.5 && key_points[4].z > 0.5){
        cv::line(bgr, cv::Point(key_points[2].x, key_points[2].y), cv::Point(key_points[4].x, key_points[4].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[5].z > 0.5 && key_points[6].z > 0.5){

        cv::line(bgr, cv::Point(key_points[0].x, key_points[0].y), neck, cv::Scalar(127,255,0), 2);
        cv::line(bgr, neck, cv::Point(key_points[5].x, key_points[5].y), cv::Scalar(127,255,0), 2);
        cv::line(bgr, neck, cv::Point(key_points[6].x, key_points[6].y), cv::Scalar(127,255,0), 2);

        if(key_points[11].z > 0.5){
            cv::line(bgr, neck, cv::Point(key_points[11].x, key_points[11].y), cv::Scalar(127,255,0), 2);
        }
        if(key_points[12].z > 0.5){
            cv::line(bgr, neck, cv::Point(key_points[12].x, key_points[12].y), cv::Scalar(127,255,0), 2);
        }
    }

    if(key_points[5].z > 0.5 && key_points[7].z > 0.5){
        cv::line(bgr, cv::Point(key_points[5].x, key_points[5].y), cv::Point(key_points[7].x, key_points[7].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[7].z > 0.5 && key_points[9].z > 0.5){
        cv::line(bgr, cv::Point(key_points[7].x, key_points[7].y), cv::Point(key_points[9].x, key_points[9].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[6].z > 0.5 && key_points[8].z > 0.5){
        cv::line(bgr, cv::Point(key_points[6].x, key_points[6].y), cv::Point(key_points[8].x, key_points[8].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[8].z > 0.5 && key_points[10].z > 0.5){
        cv::line(bgr, cv::Point(key_points[8].x, key_points[8].y), cv::Point(key_points[10].x, key_points[10].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[8].z > 0.5 && key_points[10].z > 0.5){
        cv::line(bgr, cv::Point(key_points[8].x, key_points[8].y), cv::Point(key_points[10].x, key_points[10].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[11].z > 0.5 && key_points[13].z > 0.5){
        cv::line(bgr, cv::Point(key_points[11].x, key_points[11].y), cv::Point(key_points[13].x, key_points[13].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[13].z > 0.5 && key_points[15].z > 0.5){
        cv::line(bgr, cv::Point(key_points[13].x, key_points[13].y), cv::Point(key_points[15].x, key_points[15].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[12].z > 0.5 && key_points[14].z > 0.5){
        cv::line(bgr, cv::Point(key_points[12].x, key_points[12].y), cv::Point(key_points[14].x, key_points[14].y), cv::Scalar(127,255,0), 2);
    }
    if(key_points[14].z > 0.5 && key_points[16].z > 0.5){
        cv::line(bgr, cv::Point(key_points[14].x, key_points[14].y), cv::Point(key_points[16].x, key_points[16].y), cv::Scalar(127,255,0), 2);
    }

    for(auto& kp: key_points){
        if(kp.z > 0.5){
            cv::circle(bgr, cv::Point(kp.x,kp.y), 3, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        }
    }
}
