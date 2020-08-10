
// std
#include <iostream>
#include <random>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// project
#include "quilting.hpp"
#include "nnf.hpp"
#include "recon.hpp"

using namespace cv;
using namespace std;

// main program
//
int main( int argc, char** argv ) {
    cout << argc << endl;
    // check we have exactly 2 additional arguments
    // eg. res/vgc-logo.png
    if( argc != 4) {
        cerr << "Argument error" << endl;
        abort();
    }
    
    // read the file
    Mat sourceImg, targetImg, quiltImg;
    sourceImg = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    targetImg = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    quiltImg = imread(argv[3], CV_LOAD_IMAGE_COLOR);
    
    // check for invalid input
    if(!sourceImg.data || !targetImg.data) {
        cerr << "Could not open or find the image" << std::endl;
        abort();
    }
    
    Mat bestMatchPatch;
    
    Mat nnf(targetImg.rows, targetImg.cols, CV_32SC2, Scalar(0,0));
    Mat dist(nnf.rows, nnf.cols, CV_32F, Scalar(0,0));
    
    Mat targetExtend, sourceExtend, quiltExtend;
    copyMakeBorder(sourceImg, sourceExtend, 3, 3, 3, 3, BORDER_CONSTANT);
    copyMakeBorder(targetImg, targetExtend, 3, 3, 3, 3, BORDER_CONSTANT);
    copyMakeBorder(quiltImg, quiltExtend, 0, 0, 0, 100, BORDER_CONSTANT);
    
    
    initialize(sourceImg, nnf, dist, targetImg, sourceExtend, targetExtend);
    cout<<"initialised"<<endl;
    
    for(int k = 0; k < 5; k++){
        if( k % 2 == 0){
            cout<<"front propagate"<<endl;
        bPropagate(sourceImg, targetImg, dist, nnf, sourceExtend, targetExtend);
        }
        else{
            cout<<"back propagate"<<endl;
        fPropagate(sourceImg, targetImg, dist, nnf, sourceExtend, targetExtend);
        }
        cout<<"random search"<<endl;
        randomSearch(sourceImg, targetImg, nnf, sourceExtend, targetExtend, dist);
    }
    Mat output, output2, output3, output4;
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator(seed);
      std::uniform_int_distribution dis(0, 99);
      int rowOffset = dis(generator);
      int colOffset = dis(generator);
    Mat randomPatch(quiltImg, Rect(colOffset, rowOffset, 100, 100));
    
    cout<<"quilt 1"<<endl;
    quilt(quiltImg, randomPatch, bestMatchPatch, quiltExtend);
    hconcat(randomPatch, bestMatchPatch, output);
    cout<<"quilt 2"<<endl;
    Mat bestMatchPatch2;
    quilt(quiltImg, bestMatchPatch, bestMatchPatch2, quiltExtend);
    hconcat(output, bestMatchPatch2, output2);
    cout<<"quilt 3"<<endl;
    Mat bestMatchPatch3;
    quilt(quiltImg, bestMatchPatch, bestMatchPatch3, quiltExtend);
    hconcat(output2, bestMatchPatch3, output3);
    cout<<"quilt 4"<<endl;
    Mat bestMatchPatch4;
    quilt(quiltImg, bestMatchPatch, bestMatchPatch4, quiltExtend);
    hconcat(output3, bestMatchPatch4, output4);
    
    Mat nnfImg = nnf2img(nnf, sourceImg);
    Mat recon = reconstruct(sourceImg, nnf);
    
    
  
    // save image
    imwrite("/output/CoreNNF.png", nnfImg);
    imwrite("/output/CoreReconstruction.png", recon);
    imwrite("/output/Completiom.png", output4);
    //imwrite("/output/CoreNNF.png", nnfImg);
    //imwrite("/output/CoreReconstruction.png", recon);
    
    
    string img_display_2 = "NNFImage";
    namedWindow(img_display_2, WINDOW_AUTOSIZE);
    imshow(img_display_2, nnfImg);
    
    string img_display_3 = "reconImage";
    namedWindow(img_display_3, WINDOW_AUTOSIZE);
    imshow(img_display_3, recon);
    
    string img_display_4 = "QuiltImage";
    namedWindow(img_display_4, WINDOW_AUTOSIZE);
    imshow(img_display_4, output4);
    
    // wait for a keystroke in the window before exiting
    waitKey(0);
}

