# rk3588-yolo
在香橙派max（瑞芯微3588）部署yolov5进行实时目标检测
备忘录：yolov5部署香橙派max（pt->onnx->rknn），单一简单目标识别40帧

TODO:转kmodel，rk3588实现yolo加速，高速读流与推流

## 相关链接

[airockchip/yolov5源文件](https://github.com/airockchip/yolov5)

[rknn-toolkit2编译工具链](https://github.com/airockchip/rknn-toolkit2)

[CSDN pt转rknn指南](https://blog.csdn.net/weixin_51651698/article/details/130187558?sharetype=blog&shareId=130187558&sharerefer=APP&sharesource=2302_80007495&sharefrom=qq)

[CSDN YOLOv5普通训练](https://blog.csdn.net/m0_62237233/article/details/127328106?ops_request_misc=%257B%2522request%255Fid%2522%253A%25227cfc9d45cb23ce402ef8766c363d0c22%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=7cfc9d45cb23ce402ef8766c363d0c22&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-127328106-null-null.142^v102^pc_search_result_base1&utm_term=%E5%B0%8F%E7%99%BDYOLOv5%E5%85%A8%E6%B5%81%E7%A8%8B-%E8%AE%AD%E7%BB%83%2B%E5%AE%9E%E7%8E%B0%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB&spm=1018.2226.3001.4187)

> [!WARNING] 
>
> 1.部署rknn需要运行的的yolov5源码和yolov5官方的源码有区别，别混淆
>
> 2.rknn-toolkit2一般部署在linux，网上也有虚拟机教程

## 详细

1. ### 环境

   建议使用anconda搭建Python运行的虚拟环境，根据显卡选择anconda的版本

   ```python
   #Conda基本命令
   conda create --name Pytorch	#创建名为Pytorch的新虚拟环境
   conda env list				#查看已有环境
   conda activate Pytorch		#切换到名为Pytorch的虚拟环境
   conda list					#查看环境包
   conda deactivate			#退出虚拟环境
   ```

   > [!WARNING] 
   >
   > Windows中，请将Anaconda添加至系统环境变量,否则小心powershell无法识别conda命令，或修改Vscode/Pycharm终端启动文件为cmd

   ```python
   #打开yolov5源文件，配置Pycharm解释器为刚创建的Pytorch，运行命令安装依赖包
   pip install -r requirements.txt
   ```

   环境配置仍有报错则根据报错自行查找问题

   

2. 在yolov5-master文件夹新建VOCData文件夹存放数据，其下新建Annotations文件夹存放xml标签，images文件夹存放图片（都有的话就不用新建了），将原始数据集存放进images

3. 可使用labelimg标记数据，在\labelImg\data\predefined_classes.txt修改标签名，在labelimg修改xml存放路径为Annotations，images选取路径

4. 运行**yolov5-master\VOCData\split_train_val.py**划分数据集，无需修改文件

5. 运行**yolov5-master\VOCData\text_to_yolo.py**将.xml文件转为.txt文件 ，需要修改**classes**和底部一系列的**文件路径**，生成 labels 文件夹和 dataSet_path 文件夹

6. 修改**yolov5-master\data\myvoc.yaml**，修改文件路径，**nc（标签种类数）**，**names（标签名）**

7. 确保已有**yolov5-master\VOCData\kmeans.py**的情况下，修改**clauculate_anchors.py**中FILE_ROOT，ANCHORS_TXT_PATH，CLASS_NAMES后运行，生成anchor.txt

8. 将**yolov5-master\VOCData\anchor.txt**中**Best Anchors**锚点四舍五入后放入**yolov5-master\models\yolov5s.yaml**（前提是确保训练程序参数选用的是yolov5s），同时更改**nc**，

9. 权重文件我已将添加至**yolov5-master\weights\yolov5s.pt**，无需再下载

10. 打开**yolov5-master**的终端，运行`python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 200 --batch-size 8 --img 640  --device 0 --workers 0`

       > [!WARNING] 
       >
       > 内存不够就降低batch-size，echch在200轮即可，过低降低准确率，过高会过拟合

11. 训练完成后会在**yolov5-master\runs\train**生成新的exp文件，将best.pt置于yolov5-master下

12. (可忽略）在detect.py可检验模型效果，需要修改训练好的.pt文件，测试图片，myvoc.yaml文件

    ```python
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp3/weights/best.pt", help="model path or 	triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "VOCData/images/1_1.jpg", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/myvoc.yaml", help="(optional) dataset.yaml path")
    ```

13. 打开**yolov5-master\models\yolo.py**，发现有两个forward函数。需要转pt为onnx时，应该注释下面的forward函数，打开上面的forward函数。

    > [!WARNING] 
    >
    > 需要训练时，记得将上面的forward函数注释掉，打开下面的forward函数（详情见CSDN图）

14. 复制**best.pt**到yolov5-master（转成onnx完记得删除），终端打开yolov5-master，运行`python export.py --rknpu --weights best.pt --img 640 --batch 1 --include onnx --opset 12`，运行后得到**best.onnx**和**RK_anchors.txt**

	 > [!WARNING] 
     >
     > yolo官方的源码没有添加--rknpu参数，会报错，不会生成RK_anchors.txt。

16. linux端

17. rk端
