# 项目概况
这是一个有关无人机航拍图像拼接的项目。\
特征检测与描述方法：surf。\
匹配算法是，首先采用knnMatch剔除最近匹配点距离与次近匹配点距离比率大于0.6的舍去，然后双向暴力匹配，选出互为对方最佳的匹配点的匹配，这样可以极大减少错误匹配。

# 项目文件目录结构
include 头文件 \
src     源文件 \
src_image  放置待拼接的图像 \
result_image 放置拼接后的全景图，以及展示拼接整个过程的图 

# 项目运行方法
首先执行configure.sh脚本进行编译，然后可执行文件会出现在build中。\
然后直接./main便可运行，运行时要注意的是： 
1. tmp res 图像展示拼接过程，每按一次键盘上的字母“n”，便向全景图中新拼入一张照片，但到15,16及以后时，可能速度较慢，请耐心等待。 
2. 当出现res图像框时，代表完成所有图像的拼接。这是可继续按键盘上的“n”，快速的无延迟的看拼接的整个过程。

# 运行结果
全景图放在了result_image文件夹中，processImage放在了result_image/processImage文件夹中。

# MyStitch类接口说明
```cpp
bool StitchPano(vector<cv::Mat> &images,cv::Mat &pano);
```
    这是开放的拼接图像的接口，pano为结果，images为传入的Mat向量。
```cpp
void GetProcessImages(vector<cv::Mat> &processImages);
```

    这是开放的获取过程图像的接口，processImages即为结果。