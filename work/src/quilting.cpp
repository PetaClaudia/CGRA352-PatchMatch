
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
#include "quilting.hpp"

void quilt(Mat &quiltImg, Mat &randomPatch, Mat &bestMatchPatch, Mat &quiltExtend){
    //    1.Randomly choose a 100x100 pixel-sized patch as the beginning of the synthesis.
    //    2.Find a patch to overlap the current synthesis by 20 pixels on its right. The patch should minimize the difference in the overlapping region using SSD.
    //    3.Use dynamic programmingto find the best seam from the top edge to the bottom edge.
    //    4.Repeat steps2-3 until the synthesis is 500 pixels wide(5 iterations).
    
  
    //cout<<"quiltImg rows cols "<<quiltImg.rows<<"  "<<quiltImg.cols<<endl;
    //cout<<rowOffset<<"  "<<colOffset<<endl;
    
    float ssd = numeric_limits<float>::infinity();
    Mat testPatch;
    
    
    //the 20 pixel overlap area
    for (int i = 0; i< randomPatch.rows; i++){
        for (int j = randomPatch.cols-20; j< randomPatch.cols; j++){
            //cout<<"1"<<endl;
            //get pixel in overlap reigon
            Vec2i pixel = quiltImg.at<Vec2i>(i, j);
            int x = j-3;
            int y = i-3;
            if(y>3 && y <= quiltImg.rows-3 && x>3 && x <= quiltImg.cols-3 ){
            Mat tp(quiltExtend, Rect(x, y, 100, 100));
                testPatch = tp.clone();
            }
            else{
               Mat tp(quiltExtend, Rect(j, i, 100, 100));
                testPatch = tp.clone();
            }
        
    
            //the area in origianal image that can be patched
            for (int i = 0; i< quiltImg.rows-100; i++){
                for (int j = 0; j<quiltImg.cols-100; j++){
                     //cout<<"2"<<endl;
                    Mat overlapPatch(quiltImg, Rect(j, i, 100, 100));
                    float newSSD = norm(testPatch, overlapPatch);
                     //cout<<"newSSD "<<newSSD<<endl;
                    if(newSSD<ssd){
                       //  cout<<"3"<<endl;
                        ssd=newSSD;
                        bestMatchPatch = overlapPatch.clone();
                        
                    }
                }
            }
        }
    }
    
    
}
