//
//  nnf.cpp
//  cgra352
//
//  Created by Peta Douglas on 30/04/20.
//

// std
#include <iostream>
#include <random>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <chrono>
using namespace cv;
using namespace std;
#include "nnf.hpp"


//convert offset coordinates to color image
//assumes nnf stores a Matrix of Vec2i(i,j)
//absolute positions into the source of size s
cv::Mat nnf2img(cv::Mat nnf, cv::Mat s){
    cv::Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, cv::Scalar(0,0,0));
    for(int i=0; i<nnf.rows; ++i){
        for(int j=0; j<nnf.cols; ++j){
            Vec2i p = nnf.at<Vec2i>(i,j);
            if(p[0]<0||p[1]<0||p[0]>=s.rows||p[1]>=s.cols){
                /*coordinate is outside, insert error of choice*/
                cout<<"outside range"<<endl;
            }
            int r = int(p[1]*255.0/s.cols);//cols->red
            int g = int(p[0]*255.0/s.rows);//rows->green
            int b = 255 - max(r,g);//blue
            nnf_img.at<Vec3b>(i,j)=Vec3b(b,g,r);
        }
    }
    return nnf_img;
}

void initialize(const Mat &sourceImg, Mat &nnf, Mat &dist, Mat &targetImg, Mat &sourceExtend, Mat &targetExtend){
    srand(time(NULL));
    for (int i = 0; i < nnf.rows; i++) {
        for (int j = 0; j < nnf.cols; j++) {
            int rowOffset = rand() % sourceImg.rows;
            int colOffset = rand() % sourceImg.cols;
            Mat targetPatch, sourcePatch;
            nnf.at<Vec2i>(i, j) = Vec2i(rowOffset, colOffset);
            
            if(i < 3 || i >= nnf.rows-3 || j >= nnf.cols-3 || j < 3){
                Mat tp(targetExtend, Rect(j, i, 7, 7));
                targetPatch = tp.clone();
            }
            else{
                Mat tp(targetImg, Rect(j-3, i-3, 7, 7));
                targetPatch = tp.clone();
            }
            if(colOffset < 3 || colOffset >= nnf.cols-3 || rowOffset < 3 || rowOffset >= nnf.rows-3){
                Mat rp(sourceExtend, Rect(colOffset, rowOffset, 7, 7));
                sourcePatch = rp.clone();
            }
            else{
                Mat sp(sourceImg, Rect(colOffset-3, rowOffset-3, 7, 7));
                sourcePatch = sp.clone();
            }
            dist.at<float>(i, j) = norm(sourcePatch, targetPatch);
        }
    }
}

void randomSearch(Mat &sourceImg, Mat &targetImg, Mat &nnf, Mat &sourceExtend, Mat &targetExtend, Mat &dist){
    for(int i = 0; i < nnf.rows; i++){
        for(int j = 0; j < nnf.cols; j++){
            Vec2i pixel = nnf.at<Vec2i>(i,j);
            Mat targetPatch, randomPatch;
            //set radius for next pixel
            int radius = max(sourceImg.rows, sourceImg.cols);
            //set other pixel as random numbers within the radius centered around pixel, inside image bounds
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::uniform_int_distribution dis(-radius/2,radius/2);
            int otherY = max(min(pixel[0] + dis(generator), sourceImg.rows -1), 0);
            int otherX = max(min(pixel[1] + dis(generator), sourceImg.cols -1), 0);
            while (radius >=1){
                Vec2i otherPixel(otherY, otherX);
                if(i < 3 || i >= nnf.rows-3 || j >= nnf.cols-3 || j < 3){
                    Mat tp(targetExtend, Rect(j, i, 7, 7));
                    targetPatch = tp.clone();
                }
                else{
                    Mat tp(targetImg, Rect(j-3, i-3, 7, 7));
                    targetPatch = tp.clone();
                }
                if(otherPixel[1] < 3 || otherPixel[1] >= nnf.cols-3 || otherPixel[0] < 3 || otherPixel[0] >= nnf.rows-3){
                    Mat rp(sourceExtend, Rect(otherPixel[1], otherPixel[0], 7, 7));
                    randomPatch = rp.clone();
                }
                else{
                    Mat rp(sourceImg, Rect(otherPixel[1]-3, otherPixel[0]-3, 7, 7));
                    randomPatch = rp.clone();
                }
                float ssd = norm(targetPatch, randomPatch);
                //if neighbour ssd is smaller, set pixel as other, set new ssd
                if(ssd < dist.at<float>(i,j)){
                    dist.at<float>(i,j) = ssd;
                    nnf.at<Vec2i>(i,j) = otherPixel;
                }
                radius/=2;
            }
        }
    }
}

void fPropagate(Mat &sourceImg, Mat &targetImg, Mat &dist, Mat &nnf, Mat&sourceExtend, Mat&targetExtend) {
    for (int i = 0; i <nnf.rows; i ++) {
        for (int j = 0; j < nnf.cols; j ++) {
            Vec2i pixel = nnf.at<Vec2i>(i,j);
            Mat targetPatch, neighbourPatch;
            if(i < 3 || i >= nnf.rows-3 || j >= nnf.cols-3 || j < 3){
                Mat tp(targetExtend, Rect(j, i, 7, 7));
                targetPatch = tp.clone();
            }
            else{
                Mat tp(targetImg, Rect(j-3, i-3, 7, 7));
                targetPatch = tp.clone();
            }
            //propagate left
            if (j-1 >= 0 && j-1 < nnf.cols) {
                Vec2i leftPixel = nnf.at<Vec2i>(i, j-1) + Vec2i(0,1);
                //check inside image
                if(leftPixel[1]<0||leftPixel[0]>=nnf.rows||leftPixel[1]>=nnf.cols){
                    //cout<<"not setting left neighbour"<<endl;
                    leftPixel += Vec2i(0,-1);
                }
                if(leftPixel[1] < 3 || leftPixel[1] >= nnf.cols-3 || leftPixel[0] < 3 || leftPixel[0] >= nnf.rows-3){
                    Mat rp(sourceExtend, Rect(leftPixel[1], leftPixel[0], 7, 7));
                    neighbourPatch = rp.clone();
                }
                else{
                    Mat rp(sourceImg, Rect(leftPixel[1]-3, leftPixel[0]-3, 7, 7));
                    neighbourPatch = rp.clone();
                }
                
                float ssd = norm(neighbourPatch, targetPatch);
                //if neighbour ssd is smaller, set pixel as neighbour, set new ssd
                if(ssd < dist.at<float>(i,j)){
                    nnf.at<Vec2i>(i, j) = leftPixel;
                    dist.at<float>(i, j) = ssd;
                }
            }
            //propagate up
            if (i-1 >= 0 && i-1 < nnf.rows) {
                Vec2i upPixel = nnf.at<Vec2i>(i-1, j) + Vec2i(1,0);
                //check inside image
                if(upPixel[0]<0||upPixel[1]<0||upPixel[0]>=nnf.rows||upPixel[1]>=nnf.cols){
                    upPixel += Vec2i(-1, 0);
                }
                if(upPixel[1] < 3 || upPixel[1] >= nnf.cols-3 || upPixel[0] < 3 || upPixel[0] >= nnf.rows-3){
                    Mat rp(sourceExtend, Rect(upPixel[1], upPixel[0], 7, 7));
                    neighbourPatch = rp.clone();
                }
                else{
                    Mat rp(sourceImg, Rect(upPixel[1]-3, upPixel[0]-3, 7, 7));
                    neighbourPatch = rp.clone();
                }
                float ssd = norm(neighbourPatch, targetPatch);
                //if neighbour ssd is smaller, set pixel as neighbour, set new ssd
                if(ssd < dist.at<float>(i,j)){
                    nnf.at<Vec2i>(i, j) = upPixel;
                    dist.at<float>(i, j) = ssd;
                }
                
            }
        }
    }
}
void bPropagate(Mat &sourceImg, Mat &targetImg, Mat &dist, Mat &nnf, Mat&sourceExtend, Mat&targetExtend) {
    for (int i = nnf.rows-1; i >= 0; i --) {
        for (int j = nnf.cols-1; j >=0; j --) {
            Vec2i pixel = nnf.at<Vec2i>(i,j);
            Mat targetPatch, neighbourPatch;
           if(i < 3 || i >= nnf.rows-3 || j >= nnf.cols-3 || j < 3){
                Mat tp(targetExtend, Rect(j, i, 7, 7));
                targetPatch = tp.clone();
            }
            else{
                Mat tp(targetImg, Rect(j-3, i-3, 7, 7));
                targetPatch = tp.clone();
            }
            //propagate right
            if (j+1 >= 0 && j+1 < nnf.cols) {
                Vec2i rightPixel = nnf.at<Vec2i>(i, j+1) + Vec2i(0, -1);
                //check inside image
                if(rightPixel[0]<0||rightPixel[1]<0||rightPixel[0]>=nnf.rows||rightPixel[1]>=nnf.cols){
                    rightPixel += Vec2i(0, 1);
                }
                if(rightPixel[1] < 3 ||  rightPixel[1] >= nnf.cols-3 || rightPixel[0] < 3 || rightPixel[0] >= nnf.rows-3){
                    Mat rp(sourceExtend, Rect(rightPixel[1], rightPixel[0], 7, 7));
                    neighbourPatch = rp.clone();
                }
                else{
                    Mat rp(sourceImg, Rect(rightPixel[1]-3, rightPixel[0]-3, 7, 7));
                    neighbourPatch = rp.clone();
                }
                float ssd = norm(neighbourPatch, targetPatch);
                
                //if neighbour ssd is smaller, set pixel as neighbour, set new ssd
                if(ssd < dist.at<float>(i,j)){
                    nnf.at<Vec2i>(i, j) = rightPixel;
                    dist.at<float>(i, j) = ssd;
                }
            }
            //propagate down
            if (i+1 >= 0 && i+1 < nnf.rows) {
                Vec2i downPixel = nnf.at<Vec2i>(i+1, j) + Vec2i(-1, 0);
                //check inside image
                if(downPixel[0]<0||downPixel[1]<0||downPixel[0]>=nnf.rows||downPixel[1]>=nnf.cols){
                    downPixel += Vec2i(1, 0);
                }
                if(downPixel[1] < 3 || downPixel[0] < 3 || downPixel[1] >= nnf.cols-3 || downPixel[0] >= nnf.rows-3){
                    Mat rp(sourceExtend, Rect(downPixel[1], downPixel[0], 7, 7));
                    neighbourPatch = rp.clone();
                }
                else{
                    Mat rp(sourceImg, Rect(downPixel[1]-3, downPixel[0]-3, 7, 7));
                    neighbourPatch = rp.clone();
                }
                float ssd = norm(neighbourPatch, targetPatch);
                //if neighbour ssd is smaller, set pixel as neighbour, set new ssd
                if(ssd < dist.at<float>(i,j)){
                    nnf.at<Vec2i>(i, j) = downPixel;
                    dist.at<float>(i, j) = ssd;
                }
                
            }
        }
    }
}


