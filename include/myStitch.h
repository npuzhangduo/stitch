#include <iostream>  
#include <stdio.h>  
#include "opencv2/core.hpp"  
#include "opencv2/core/utility.hpp"  
#include "opencv2/core/ocl.hpp"  
#include "opencv2/imgcodecs.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/calib3d.hpp"  
#include "opencv2/imgproc.hpp"  
#include "opencv2/flann.hpp"  
#include "opencv2/xfeatures2d.hpp"  
#include "opencv2/ml.hpp"
#include <ostream>
#include <fstream>
#include <sstream>  
 
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;


struct four_corners {
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
};

class MyStitch {
public:
    MyStitch();
    ~MyStitch();
    bool StitchPano(vector<cv::Mat> &images,cv::Mat &pano);
    void GetProcessImages(vector<cv::Mat> &processImages);

    bool showStitchProcess;
    bool debug_print;
private:
    void CalcCorners(const Mat& H, const Mat& src, four_corners &corners);
    cv::Mat imageTranslation(cv::Mat & srcImage, int x0ffset, int y0ffset);
    bool wrap(vector<cv::Mat> &images,cv::Mat &pers,int number,int xoff,int yoff,int &xoffadd,int &yoffadd,cv::Mat &homa);
    vector<cv::Mat> Ho;
    vector<cv::Mat> processImages;

};