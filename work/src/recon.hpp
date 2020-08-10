//
//  Recon.hpp
//  cgra352
//
//  Created by Peta Douglas on 5/05/20.
//

#ifndef recon_hpp
#define recon_hpp

#include <stdio.h>

// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#endif /* recon_hpp */

cv::Mat reconstruct(Mat &sourceImg, Mat &nnf);
