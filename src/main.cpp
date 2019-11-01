#include "myStitch.h"
using namespace std;


int main(int argc, char const *argv[])
{

// 读入图片
    int num = 20; //用来控制读取图片的数量
	vector<cv::Mat> images;

	//按顺序读入
	for (int i = 0;i <= num;i++) {
		stringstream ss;
        ss << i;
        string str = ss.str();
		for (int k = str.length();k < 4;k++) {
			str = '0' + str;
		}
        string filename = "../src_image/dataset1/" + str + ".bmp";
		cv::Mat image = cv::imread(filename);
		cv::Mat tmp;
		cv::resize(image,tmp,cv::Size(image.cols / 2,image.rows / 2));
		images.push_back(tmp);
	}

	cv::Mat pano;
	vector<cv::Mat> processImages;

	// 新建拼接类，默认展示拼接过程
	MyStitch* test = new MyStitch();
    test->StitchPano(images,pano);
	test->GetProcessImages(processImages);

	cv::imshow("res",pano);
	cv::imwrite("../result_image/panorama_result.jpg",pano);

	
	for (int i = 0;i < processImages.size();i++) {
		cv::imshow("process",processImages[i]);
		stringstream ss;
        ss << i;
        string str = ss.str();
		string filename = "../result_image/processImage/" + str + ".jpg";
		cv::imwrite(filename,processImages[i]);
		char c = waitKey();
		
		while (c != 'n' && c != 'N' && c != 'p' && c != 'P');
		if (c == 'p' || c == 'P') i = i - 2;
	}
 
	waitKey();

    return 0;
}
