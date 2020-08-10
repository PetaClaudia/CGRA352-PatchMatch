//
//  nnf.hpp
//  cgra352
//
//  Created by Peta Douglas on 30/04/20.
//

#ifndef nnf_hpp
#define nnf_hpp

#include <stdio.h>

// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
#endif /* nnf_hpp */

cv::Mat nnf2img(cv::Mat nnf, cv::Mat s);
void initialize(const Mat &sourceImg, Mat &nnf, Mat &dist, Mat & targetImg, Mat &sourceExtend, Mat &targetExtend);
void randomSearch(Mat &sourceImg, Mat &targetImg, Mat &nnf, Mat &sourceExtend, Mat &targetExtend, Mat &dist);
void fPropagate(Mat &sourceImg, Mat &targetImg, Mat &dist, Mat &nnf, Mat &sourceExtend, Mat &targetExtend);
void bPropagate(Mat &sourceImg, Mat &targetImg, Mat &dist, Mat &nnf, Mat &sourceExtend, Mat &targetExtend);
