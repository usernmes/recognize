import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox,
    QProgressBar, QSizePolicy, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# 加载 YOLO 模型
try:
    model = YOLO('D:/深度大作业/runs/train/face_expression_model/weights/best.pt')
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit()

# 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

# 表情类别映射到中文
class_name_to_chinese = {
    "Angry": "生气",
    "Disgust": "厌恶",
    "Fear": "害怕",
    "Happy": "高兴",
    "Sad": "伤心",
    "Surprise": "惊讶",
    "Neutral": "中性"
}

# 使用 PIL 在图像上绘制中文文字
def draw_text_chinese(image, text, position, font_path="simsun.ttc", font_size=20, color=(0, 255, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 视频处理线程
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(str)

    def __init__(self, video_path, output_dir, parent=None):
        super(VideoThread, self).__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        self.paused = False
        self.out = None
        self.frame_buffer = []  # 缓存已处理的视频帧
        self.processed_frame_interval = 5  # 每隔5帧处理一次

    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_dir, "output_video.mp4")
        self.out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

        while self.running and self.cap.isOpened():
            if self.paused:
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            self.progress_signal.emit(progress)

            if frame_count % self.processed_frame_interval == 0:  # 每隔几帧处理一次
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                detected_classes = []
                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_resized = face_pil.resize((48, 48))
                    temp_path = "temp_face.jpg"
                    face_resized.save(temp_path)

                    results = model(temp_path)
                    result = results[0]

                    cls_indices = result.boxes.cls
                    confidences = result.boxes.conf
                    class_names = model.names

                    if len(cls_indices) > 0:
                        class_id = int(cls_indices[0])
                        class_name = class_names[class_id]
                        class_name_chinese = class_name_to_chinese.get(class_name, "未知")
                        confidence = confidences[0]
                        detected_classes.append((class_name_chinese, confidence, (x, y, w, h)))

                # 绘制检测结果并保存视频帧
                for cls_name, conf, (x, y, w, h) in detected_classes:
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = draw_text_chinese(frame, label, (x, y - 30))

                self.frame_buffer.append(frame)  # 将处理后的帧存入缓冲区

                # 更新结果文本
                result_text = "\n".join([f"{cls_name}: {conf:.2f}" for cls_name, conf, _ in detected_classes])
                self.result_signal.emit(result_text)

            # 如果缓存中有帧，发送到主线程
            if self.frame_buffer:
                frame_to_show = self.frame_buffer.pop(0)
                self.frame_signal.emit(frame_to_show)

            # 检查是否播放到最后一帧
            if frame_count == total_frames:
                self.running = False
                self.progress_signal.emit(100)  # 设置进度条为 100%
                self.result_signal.emit("视频播放完毕！")  # 播放完毕提示
            # self.out.write(frame)  # 写入视频

        self.cap.release()
        # self.out.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("人脸表情识别系统")
        self.setGeometry(100, 100, 1000, 700)  # 调整窗口大小
        self.setMaximumSize(1200, 800)  # 设定最大宽度和最大高度

        # 全局变量
        self.image_path = None
        self.video_path = None
        self.video_thread = None

        # 主窗口布局
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # 使用横向布局

        # 左侧区域：显示图像或视频
        left_layout = QVBoxLayout()
        self.image_label = QLabel("展示区域")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.image_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # 右侧区域：按钮和识别结果
        right_layout = QVBoxLayout()

        # 添加预测结果的标题
        self.result_label = QLabel("预测结果")
        self.result_label.setAlignment(Qt.AlignCenter)  # 设置居中对齐
        right_layout.addWidget(self.result_label)

        # 调整预测结果文本框宽度
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)  # 设置为只读
        self.result_text_edit.setMaximumHeight(400)  # 设置最大高度
        self.result_text_edit.setFixedWidth(250)  # 设置固定宽度，避免过宽
        right_layout.addWidget(self.result_text_edit)

        # 按钮布局
        button_layout = QVBoxLayout()  # 使用垂直布局

        # 调整按钮宽度
        button_width = 250  # 设置按钮宽度

        self.choose_image_button = QPushButton("选择图片")
        self.choose_image_button.setFixedWidth(button_width)  # 设置按钮固定宽度
        self.choose_image_button.clicked.connect(self.choose_image)
        button_layout.addWidget(self.choose_image_button)

        self.detect_image_button = QPushButton("检测图片")
        self.detect_image_button.setFixedWidth(button_width)  # 设置按钮固定宽度
        self.detect_image_button.clicked.connect(self.detect_faces_and_expressions)
        button_layout.addWidget(self.detect_image_button)

        self.choose_video_button = QPushButton("选择视频")
        self.choose_video_button.setFixedWidth(button_width)  # 设置按钮固定宽度
        self.choose_video_button.clicked.connect(self.choose_video)
        button_layout.addWidget(self.choose_video_button)

        self.start_video_button = QPushButton("播放视频")
        self.start_video_button.setFixedWidth(button_width)  # 设置按钮固定宽度
        self.start_video_button.clicked.connect(self.start_video)
        button_layout.addWidget(self.start_video_button)

        self.pause_video_button = QPushButton("暂停/继续视频")
        self.pause_video_button.setFixedWidth(button_width)  # 设置按钮固定宽度
        self.pause_video_button.clicked.connect(self.toggle_pause_video)
        button_layout.addWidget(self.pause_video_button)

        right_layout.addLayout(button_layout)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def choose_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def choose_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def detect_faces_and_expressions(self):
        """检测图片中的人脸并预测表情"""
        if not self.image_path:
            QMessageBox.critical(self, "错误", "请先选择一张图片！")
            return

        try:
            img = cv2.imread(self.image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                QMessageBox.information(self, "结果", "未检测到任何人脸！")
                return

            detected_classes = []
            for (x, y, w, h) in faces:
                face_img = img[y:y + h, x:x + w]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_resized = face_pil.resize((48, 48))
                temp_path = "temp_face.jpg"
                face_resized.save(temp_path)

                results = model(temp_path)
                result = results[0]

                cls_indices = result.boxes.cls
                confidences = result.boxes.conf
                class_names = model.names

                if len(cls_indices) > 0:
                    class_id = int(cls_indices[0])
                    class_name = class_names[class_id]
                    class_name_chinese = class_name_to_chinese.get(class_name, "未知")
                    confidence = confidences[0]
                    detected_classes.append((class_name_chinese, confidence, (x, y, w, h)))

            for cls_name, conf, (x, y, w, h) in detected_classes:
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = draw_text_chinese(img, label, (x, y - 30))

            output_dir = "D:/深度大作业/outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(self.image_path))
            cv2.imwrite(output_path, img)

            self.display_image(output_path)

            result_text = "\n".join([f"{cls_name}: {conf:.2f}" for cls_name, conf, _ in detected_classes])
            self.result_text_edit.setText(f"检测完成！\n{result_text}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {e}")

    def start_video(self):
        if not self.video_path:
            QMessageBox.critical(self, "错误", "请先选择一个视频文件！")
            return

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        output_dir = "D:/深度大作业/outputs"
        self.video_thread = VideoThread(self.video_path, output_dir)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.progress_signal.connect(self.update_progress)
        self.video_thread.result_signal.connect(self.update_result)
        self.video_thread.start()

    def toggle_pause_video(self):
        if self.video_thread:
            self.video_thread.toggle_pause()

    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def update_result(self, result_text):
        self.result_text_edit.setText(result_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())