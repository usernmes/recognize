# 基于Yolov8+fer2013.csv的人脸表情识别系统 （训练时间较长，请自行安排时间，且电脑要有GPU，显存不低于4GB）
# ！注意，本项目所需的fer2013.csv,yolov8x.pt需自己准备，具体环境参考requirements.txt
首先运行“创建目录.py”，文件位置可以自行选择，但后续需要自己更改。

设备测试：运行设备测试.py，查看torch是否符合
<img width="244" height="48" alt="image" src="https://github.com/user-attachments/assets/19fcfd80-b090-4918-ac4d-fc01791cb5a6" />

数据准备：自行下载fer2013.csv文件，将其放到指定目录，可自行指定，运行数据准备.py，运行结果如下

<img width="969" height="280" alt="image" src="https://github.com/user-attachments/assets/8709aff9-8c1c-4806-a7f5-b117dc85b02e" />

训练模型：自行下载yolov8x.pt文件，放到指定位置，运行训练模型.py

<img width="969" height="371" alt="image" src="https://github.com/user-attachments/assets/9f4d4114-be72-4131-8ad5-8abbcda2abc4" />
<img width="969" height="345" alt="image" src="https://github.com/user-attachments/assets/64ced0a8-7f32-4178-a4a8-9c41084af9f1" />
<img width="969" height="311" alt="image" src="https://github.com/user-attachments/assets/40d80d46-c912-46a6-bb43-e1300dbbda88" />
<img width="969" height="372" alt="image" src="https://github.com/user-attachments/assets/821afa9d-ea26-4bd8-8c7d-6f2117791cca" />
<img width="969" height="378" alt="image" src="https://github.com/user-attachments/assets/9fdc76f3-afde-4bf5-9153-17e5084a174e" />
<img width="969" height="329" alt="image" src="https://github.com/user-attachments/assets/1b88b361-9d51-44a0-a6ec-57362cc4026c" />

模型测试： 运行模型测试.py，检测测试结果
<img width="980" height="332" alt="image" src="https://github.com/user-attachments/assets/e30ef8a7-5427-4b8b-be11-8148a88b2039" />

窗口识别：运行识别窗口.py，打开一个pyqt5的界面，界面提供选项，检测图片或者检测视频。
<img width="974" height="418" alt="image" src="https://github.com/user-attachments/assets/4eb6c18a-52d9-443f-9ddb-db5f2b43751c" />
<img width="974" height="470" alt="image" src="https://github.com/user-attachments/assets/f0e16171-4449-4fc3-a7ef-8be2a55e344d" />
<img width="974" height="495" alt="image" src="https://github.com/user-attachments/assets/40e9f1e8-db20-4077-bf7f-9d9eb7c0c3ad" />

训练集（图片及标签）位置如图所示
 <img width="969" height="351" alt="image" src="https://github.com/user-attachments/assets/216e1d65-449b-4902-bb1d-7246e456e89c" />

测试集（图片及标签）位置如图所示
 <img width="969" height="403" alt="image" src="https://github.com/user-attachments/assets/08207b81-1197-4eca-b028-68b4c8c7e0fb" />

模型保存位置如图所示
 <img width="969" height="671" alt="image" src="https://github.com/user-attachments/assets/fff828d2-7238-432b-b952-c29c4854d9ed" />

模型测试结果位置如图所示
 <img width="969" height="581" alt="image" src="https://github.com/user-attachments/assets/27249e2e-09fc-47b2-b9ba-ba277f4da572" />

