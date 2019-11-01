#include "myStitch.h"

MyStitch::MyStitch() {
    debug_print = false;
    showStitchProcess = true;
}
MyStitch::~MyStitch() {
    cout <<"Finsih Stitch!!!"<<endl;
}

void MyStitch::GetProcessImages(vector<cv::Mat> &processImages) {
	processImages = this -> processImages;
}

bool MyStitch::wrap(vector<cv::Mat> &images,cv::Mat &pers,int number,int xoff,int yoff,int &xoffadd,int &yoffadd,cv::Mat &homa) {
    vector<cv::Mat> img; 
	vector< DMatch > matches_unique;
	vector<KeyPoint> key1, key2;
	cv::Mat iniHomo;
	for (int count = 0;count < number;count++) {
		Ptr<SURF> surf;
		surf = SURF::create(500);
		BFMatcher matcher0;
		BFMatcher matcher1;
		BFMatcher matcher2;
	
		cv::Mat query,train;  // descriptors
		
		surf->detectAndCompute(images[number],cv::Mat(),key1,query);
		surf->detectAndCompute(images[count],cv::Mat(),key2,train);
        if (debug_print) {
            cout <<"Key1 size:"<<key1.size()<<endl;
		    cout <<"Key2 size:"<<key2.size()<<endl;
		    cout <<"train size"<<train.size()<<endl;
		    cout <<"query size"<<query.size()<<endl;
        }
		vector< DMatch > matches;
		vector< DMatch > inv_matches;
		vector<vector<DMatch>> m_knnMatches;

		matcher0.knnMatch(query,train,m_knnMatches,2);

        if (debug_print)
		    cout <<"knnmatches"<< m_knnMatches.size() <<endl;

		for (int i=0; i<m_knnMatches.size(); i++) {
	  		if (m_knnMatches[i][0].distance / m_knnMatches[i][1].distance < 0.6) {
				matches.push_back(m_knnMatches[i][0]);
	  		}
  		}
	
	
		matcher2.match(train,query,inv_matches);
        if (debug_print) {
            cout <<"matches"<<matches.size()<<endl;
		    cout <<"inv_matches"<<inv_matches.size()<<endl;
        }
	    vector< DMatch > matches_unique_t;
		for (int i = 0;i < matches.size();i++) {
			for (int j = 0;j < inv_matches.size();j++) {
				if (inv_matches[j].queryIdx == matches[i].trainIdx) {
					if (inv_matches[j].trainIdx == matches[i].queryIdx) {
						matches_unique_t.push_back(matches[i]);
					}
				}
			}
		}

		double all = 0;
		for (int i = 0;i < matches_unique_t.size();i++) {
			all += matches_unique_t[i].distance;
		}
		double min_dis = 100000;
		for (int i = 0;i < matches_unique_t.size();i++) {
			if (matches_unique_t[i].distance < min_dis) {
				min_dis = matches_unique_t[i].distance;
			}
		}
		double max_dis = 0;
		for (int i = 0;i < matches_unique_t.size();i++) {
			if (matches_unique_t[i].distance > max_dis) {
				max_dis = matches_unique_t[i].distance;
			}
		}
        if (debug_print) {
            cout <<count<<endl;
		    cout <<"max_dis:"<<max_dis<<endl;
		    cout <<"min_dis"<<min_dis<<endl;
		    cout <<"average_dis"<<all / matches_unique_t.size()<<endl;
		    cout <<"matches_unique size:"<<matches_unique_t.size()<<endl;
		    cout <<endl;
        }
		if (matches_unique_t.size() > 200) {
			img.push_back(images[number]);
			img.push_back(images[count]);
			matches_unique = matches_unique_t;
			iniHomo = Ho[count].clone();
			break;
		}
		key1.clear();
		key2.clear();
		
	} 
    if (img.size() < 2) { 
		return false;
	}
    if (debug_print) {
        cout <<"begin cal homo:"<<matches_unique.size()<<endl;
	    cout <<"Key1:"<<key1.size()<<endl;
	    cout <<"Key2:"<<key2.size()<<endl;
    }

	sort(matches_unique.begin(),matches_unique.end());
	
	vector<DMatch> good_matches;
	
	int m = 0;
	if (matches_unique.size() > 160)
		m = 160;
	if (matches_unique.size() > 120)
		m = 120;
	else if (matches_unique.size() > 60)
		m = 60;
	else if (matches_unique.size() > 20)
		m = 20;
	else if (matches_unique.size() >= 4)
		m = 4;
	else {
		cout <<"the matches is not enough"<<endl;
		return false;
	}
	
	int ptsPairs = min(m,(int)(matches_unique.size() * 0.6));

    if (debug_print)
	    cout <<"number of good matches:"<<ptsPairs<<endl;

	for (int i = 0;i < ptsPairs;i++) {
		good_matches.push_back(matches_unique[i]);
	}
	
	vector<cv::Point2f> imagePoints1, imagePoints2;

	for (int i = 0;i < good_matches.size();i++) {
		imagePoints1.push_back(key1[good_matches[i].queryIdx].pt);
		imagePoints2.push_back(key2[good_matches[i].trainIdx].pt);
	}
	
	cv::Mat homo = findHomography(imagePoints1,imagePoints2);
	
	homo = homo * iniHomo;

	homa = homo.clone();

	if (debug_print) {
        cout <<"homography matrix:"<<endl;
	    cout <<homo<<endl;
    }

	four_corners corners;
	CalcCorners(homo,img[0],corners);
	
    if (debug_print) {
        cout << "left_top:" << corners.left_top << endl;
	    cout << "left_bottom:" << corners.left_bottom << endl;
	    cout << "right_top:" << corners.right_top << endl;
	    cout << "right_bottom:" << corners.right_bottom << endl;
    }

	cv::Mat imageTransform0,dst;
	double x_offset = 0;
	double y_offset = 0;
	if (corners.left_top.x < 0 || corners.left_bottom.x < 0) {
		x_offset = -MIN(corners.left_top.x,corners.left_bottom.x);
	}
	if (corners.left_top.y < 0 || corners.right_top.y < 0) {
		y_offset = -MIN(corners.left_top.y,corners.right_top.y);
	}
	x_offset = x_offset + xoff;
	y_offset = y_offset + yoff;
    if (debug_print) {
        cout <<"x_offset"<<x_offset<<endl;
        cout <<"y_offset"<<y_offset<<endl;
    }

	imageTransform0 = imageTranslation(img[0],x_offset ,y_offset);

	int dstCols = imageTransform0.cols;
	int dstRows = imageTransform0.rows;
	
	cv::warpPerspective(imageTransform0, dst, homo, Size(dstCols,dstRows));
	pers = dst.clone();

	xoffadd = x_offset - xoff;
	yoffadd = y_offset - yoff;
	return true;
}

bool MyStitch::StitchPano(vector<cv::Mat> &images,cv::Mat &pano) {
    if (images.size() < 2) {
		cout <<"too few images"<<endl;
		return false;
	}
	int x_offset = 0;
	int y_offset = 0;
	pano = images[0].clone();
	cv::Mat A = cv::Mat::eye(3,3,CV_64F);
	Ho.push_back(A);
	processImages.push_back(pano);
	for (int i = 1;i < images.size();i++) {
		cv::Mat pers;
		int x_add = 0;
		int y_add = 0;
		cv::Mat homo;
        bool success = wrap(images,pers,i,x_offset,y_offset,x_add,y_add,homo);
		Ho.push_back(homo);
		if (success){
            if (debug_print) {
                cv::imshow("pers",pers);
			    cv::imshow("pano",pano);
            }
			for (int i = 0;i < pano.cols;i++) {
				pano.at<Vec3b>(0,i) = pano.at<Vec3b>(2,i);
				pano.at<Vec3b>(1,i) = pano.at<Vec3b>(2,i);
			}
			for (int i = 0;i < pano.rows;i++) {
				pano.at<Vec3b>(i,0) = pano.at<Vec3b>(i,1);
			}
			for (int i = 0;i < pano.cols;i++) {
				pano.at<Vec3b>(pano.rows - 1,i) = pano.at<Vec3b>(pano.rows-2,i);
			}
			for (int i = 0;i < pano.rows;i++) {
				pano.at<Vec3b>(i,pano.cols-1) = pano.at<Vec3b>(i,pano.cols-2);
			}
			pano.copyTo(pers(Rect(x_add,y_add,pano.cols,pano.rows)));
			pano = pers.clone();
			x_offset += x_add;
			y_offset += y_add;
			int xx = 0;
			int yy = 0;
			for (int i = 0;i < pano.cols;i++) {
				int flag = 0;
				for (int j = 0;j < pano.rows;j++) {
					if (pano.at<cv::Vec3b>(j, i) != cv::Vec3b(0,0,0) ) {
						xx = i;
						yy = j;
						flag = 1;
						break;
					}
				}
				if (flag) break;
			}
			pano = imageTranslation(pano,-xx,-yy);
			processImages.push_back(pano);
			x_offset -= xx;
			y_offset -= yy;
		}
        if (showStitchProcess) {
            cv::imshow("tmp res",pano);
		    char c = cv::waitKey(0);
		    if (c == 'n' || c == 'N')
			    continue;
        }
	}
}
cv::Mat MyStitch::imageTranslation(cv::Mat & srcImage, int x0ffset, int y0ffset) {
    int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat resultImage(cv::Size(srcImage.cols + x0ffset,srcImage.rows + y0ffset), srcImage.type());
	for (int i = 0; i < resultImage.rows; i++) {
		for (int j = 0; j < resultImage.cols; j++) {
			int x = j - x0ffset;
			int y = i - y0ffset;
			if (x >= 0 && y >= 0 && x < nCols && y < nRows) {
				resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
			}
			else {
				resultImage.at<cv::Vec3b>(i, j)[0] = 0;
				resultImage.at<cv::Vec3b>(i, j)[1] = 0;
				resultImage.at<cv::Vec3b>(i, j)[2] = 0;
			}
			
		}
	}
	return resultImage;
}
void MyStitch::CalcCorners(const Mat& homography, const Mat& src, four_corners &corners) {

    double v2[] = { 0, 0, 1 };
	double v1[3];
	cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);
	cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1); 
 
	V1 = homography * V2;
	corners.left_top.x = cvRound(v1[0] / v1[2]);
	corners.left_top.y = cvRound(v1[1] / v1[2]);
 
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  
	V1 = cv::Mat(3, 1, CV_64FC1, v1);
	V1 = homography * V2;
	corners.left_bottom.x = cvRound(v1[0] / v1[2]);
	corners.left_bottom.y = cvRound(v1[1] / v1[2]);
 
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  
	V1 = cv::Mat(3, 1, CV_64FC1, v1);  
	V1 = homography * V2;
	corners.right_top.x = cvRound(v1[0] / v1[2]);
	corners.right_top.y = cvRound(v1[1] / v1[2]);
 
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = cv::Mat(3, 1, CV_64FC1, v2);  
	V1 = cv::Mat(3, 1, CV_64FC1, v1); 
	V1 = homography * V2;
	corners.right_bottom.x = cvRound(v1[0] / v1[2]);
	corners.right_bottom.y = cvRound(v1[1] / v1[2]);
}