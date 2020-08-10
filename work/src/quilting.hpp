
#include <stdio.h>

// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void quilt(Mat &quiltImg, Mat &randomPatch, Mat &bestMatchPatch, Mat &quiltExtend);
