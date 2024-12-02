# Face-recognition
> 基于AI部署在低性能服务器上基于dlib与insightface的简单人脸识别

## 食用方法

```
pip install -r requirement.txt
```
[下载模型与打包的dlib](https://pan.baidu.com/s/1SaN1KtfNNlJQWCdctTD-2Q) 提取码: djit 


## 主要流程

### 预处理

​	对路径下`./Gallery`照片全部进行识别，筛选出后缀为`"jpg"`，`"png"`的格式照片进行预处理。
​	用Dlib识别照片中的人脸，使用`./shape_predictor_68_face_landmarks.dat`进行预处理，安装Dlib，可能需要编译安装，安装会相对比较久（提供了zip）。

```
pip install dlib
```

​	本仓库中提供了`extract3.py`与`extract2.py`两个版本，区别并不是很大，`extract2.py`保留了注释，`extract3.py`支持在程序因为内存不足异常中断后跳过对已经处理照片的处理（实际上如果对代码做出改写在读取文件的时候就对文件做出一定的排序，之后就可以直接从预定的切片处开始继续处理，能够极大提升代码性能）。

```
python extract2.py
python extract3.py
```



​	查询路径下文件数量

```
ls -l | wc -l
```



### 提取照片向量

​	使用`glintr100.onnx`进行人脸向量提取并且批量提取，每隔32张保存一次。

```
python get_features.py
```





### 识别上传照片

​	对上传的照片会先进行预处理，基本流程与`extract3.py`无异，接着进行向量提取，再与向量库中的特征向量进行匹配，实际上能够达到`0.5`以上基本可以认为是同个人。



## 注意

1. 本实验完全是基于ChatGPT的手把手教学，在对照片进行预处理的时候务必指导AI对照片进行细节的指导

    ```
    加载图片并转灰度图
    检测人脸
    获取关键点
    获取原始框坐标
    添加缓冲区，扩大裁剪框
    裁剪人脸
    调整到目标尺寸
    ```

​    否则AI会对只对照片格式进行裁剪，并且不使用任何模型进行人脸识别，会导致在接下来的向量提取进程中精确度大大减小，导致提取的向量是整张图片的向量，就失去了人脸识别的意义，沦为完全的照片识别。

2. 鸣谢：本次项目中参考了B站`@会AI的哈利波特`up主的源码`./源码资料/第21节：人脸关键点定位/landmark/detect_face_parts.py`与`./源码资料/第21节：人脸关键点定位/landmark/shape_predictor_68_face_landmarks.dat`

