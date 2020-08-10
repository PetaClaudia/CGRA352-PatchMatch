//
//  recon.cpp
//  cgra352
//
//  Created by Peta Douglas on 5/05/20.
//

#include "recon.hpp"

cv::Mat reconstruct(Mat &sourceImg, Mat &nnf){
    cv::Mat recon(sourceImg.rows, sourceImg.cols, sourceImg.type());
    for (int i = 0; i<sourceImg.rows; i++){
        for (int  j = 0; j<sourceImg.cols; j++){
            Vec2i coordinate = nnf.at<Vec2i>(i,j);
            recon.at<Vec3b>(i,j) = sourceImg.at<Vec3b>(coordinate);
        }
    }
    return recon;

}

