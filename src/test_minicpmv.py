from minicpm_helper import OvModelForCausalLMWithEmb, OvMiniCPMV, init_model
from minicpm_helper import lm_variant_selector
from minicpm_helper import llm_path, copy_llm_files

from PIL import Image
from pathlib import Path
import openvino as ov
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QLabel, QTextEdit, QFileDialog, QStackedWidget)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
import cv2
import numpy as np
import argparse

class InferenceThread(QThread):
    output_signal = pyqtSignal(str)
    
    def __init__(self, ov_model, image, question, tokenizer):
        super().__init__()
        self.ov_model = ov_model
        self.image = image
        self.question = question
        self.tokenizer = tokenizer
        
    def run(self):
        # 如果是视频（多帧），直接将frames和问题组合到content中
        if isinstance(self.image, list):
            msgs = [{"role": "user", "content": self.image + [self.question]}]
            image = None  # 不通过image参数传递
        else:
            # 单张图片保持原有方式
            msgs = [{"role": "user", "content": self.question}]
            image = self.image
        
        # 构造所有参数的字符串并打印到控制台
        print("\n" + "="*50)
        print("Calling ov_model.chat with parameters:")
        print(f"  image: {image}")
        print(f"  msgs: {msgs}")
        print(f"  context: {None}")
        print(f"  tokenizer: {self.tokenizer}")
        print("  sampling: False")
        print("  stream: True")
        print("  max_new_tokens: 1000")
        print("  use_image_id: False")
        print("  max_slice_nums: 2")
        print("="*50 + "\n")
        
        res = self.ov_model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=False,
            stream=True,
            max_new_tokens=1000,
            use_image_id=False,  # 添加视频处理所需的参数
            max_slice_nums=2     # 添加视频处理所需的参数
        )
        
        for new_text in res:
            self.output_signal.emit(new_text)

class CustomTextEdit(QTextEdit):
    enterPressed = pyqtSignal()
    
    def keyPressEvent(self, event):
        # Ctrl+Enter 用于换行
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            super().keyPressEvent(event)
        # Enter 用于发送消息
        elif event.key() == Qt.Key_Return:
            event.accept()  # 确保事件被接受
            self.enterPressed.emit()
        else:
            super().keyPressEvent(event)

class CustomVideoWidget(QVideoWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.media_player = None
        
    def setMediaPlayer(self, player):
        self.media_player = player
        
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.media_player:
            if self.media_player.state() == QMediaPlayer.PlayingState:
                self.media_player.pause()
            else:
                self.media_player.play()
        super().mouseDoubleClickEvent(event)

class VideoAnalyzeApp(QMainWindow):
    def __init__(self, num_frames=16, scale=1.0):
        super().__init__()
        self.ov_model = None
        self.tokenizer = None
        self.default_question = "请详细描述这张图片"
        self.num_frames = num_frames
        self.scale = scale  # 存储缩放因子
        self.setupFonts()
        self.initUI()
        self.loadModel()
        self.chat_history = []
        self.current_media_type = None
        
    def setupFonts(self):
        # 设置应用程序默认字体
        self.default_font = QFont("Microsoft YaHei", 8)
        QApplication.setFont(self.default_font)
        
        # 为不同类型的组件创建特定字体
        self.button_font = QFont("Microsoft YaHei", 8, QFont.Medium)
        self.text_font = QFont("Microsoft YaHei", 8)  # 输入输出框使用稍大的字号
        
    def initUI(self):
        self.setWindowTitle('MiniCPMV GUI')
        self.setGeometry(100, 100, 1200, 1000)
        
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 加载图片按钮
        self.load_btn = QPushButton('Load Image/Video')
        self.load_btn.setFont(self.button_font)
        self.load_btn.setMinimumHeight(48)
        self.load_btn.clicked.connect(self.loadImage)
        layout.addWidget(self.load_btn)
        
        # 创建堆叠式窗口组件来切换图片和视频显示
        self.media_stack = QStackedWidget()
        self.media_stack.setMinimumHeight(600)
        
        # 图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.media_stack.addWidget(self.image_label)
        
        # 视频播放组件
        self.video_widget = CustomVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.setMediaPlayer(self.media_player)
        
        # 添加视频状态改变的信号处理
        self.media_player.stateChanged.connect(self.handle_media_state_changed)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)
        
        self.media_stack.addWidget(self.video_widget)
        
        layout.addWidget(self.media_stack)
        
        # 输入区域
        input_layout = QHBoxLayout()
        self.input_text = CustomTextEdit()
        self.input_text.setFont(self.text_font)
        self.input_text.setMaximumHeight(48)
        self.input_text.enterPressed.connect(self.processInput)
        self.send_btn = QPushButton('Send')
        self.send_btn.setFont(self.button_font)
        self.send_btn.setMinimumHeight(48)
        self.send_btn.setMinimumWidth(64)
        self.send_btn.clicked.connect(self.processInput)
        input_layout.addWidget(self.input_text)
        input_layout.addWidget(self.send_btn)
        layout.addLayout(input_layout)
        
        # 输出区域
        self.output_text = QTextEdit()
        self.output_text.setFont(self.text_font)
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(400)
        self.output_text.setMaximumHeight(800)
        layout.addWidget(self.output_text)
        
        main_widget.setLayout(layout)
        
        # 禁用发送按钮，直到模型加载完成
        self.send_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
    def loadModel(self, device='GPU'):
        self.output_text.setText("Starting model initialization...\n")
        try:
            self.output_text.append("Initializing OpenVINO Core...")
            core = ov.Core()
            
            self.output_text.append("Loading model files...")
            model_dir = Path('../../models/minicpm_v_2_6')
            llm_int4_path = 'language_model_int4'
            
            self.output_text.append("Initializing model...")
            self.ov_model = init_model(model_dir, llm_int4_path, device)
            
            self.output_text.append("Loading tokenizer...")
            self.tokenizer = self.ov_model.processor.tokenizer
            
            self.output_text.append("Model loading completed!")
            self.send_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.output_text.append("System is ready to use!")
        except Exception as e:
            self.output_text.append(f"Error loading model: {str(e)}")

    def loadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Image/Video', '', 
            'Image/Video Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)')
        if file_name:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # 处理图片
                original_image = Image.open(file_name)
                
                # 对LLM使用的图片进行缩放
                if self.scale != 1.0:
                    new_width = int(original_image.width * self.scale)
                    new_height = int(original_image.height * self.scale)
                    self.current_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    self.current_image = original_image
                
                # GUI显示保持原样
                pixmap = QPixmap(file_name)
                scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.media_stack.setCurrentWidget(self.image_label)
                self.current_media_type = 'image'
                self.default_question = "请详细描述这张图片"
                
                # 停止视频播放
                self.media_player.stop()
                
            elif file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # 设置视频播放
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
                self.media_stack.setCurrentWidget(self.video_widget)
                # 设置为暂停状态，不自动播放
                self.media_player.pause()
                self.current_media_type = 'video'
                
                # 处理视频帧提取
                cap = cv2.VideoCapture(file_name)
                if not cap.isOpened():
                    self.output_text.append("Failed to open video file.")
                    return
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    self.output_text.append("No frames found in video.")
                    cap.release()
                    return
                
                indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # 将 BGR 转为 RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame)
                        
                        # 对每一帧进行缩放
                        if self.scale != 1.0:
                            new_width = int(pil_frame.width * self.scale)
                            new_height = int(pil_frame.height * self.scale)
                            pil_frame = pil_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                        frames.append(pil_frame)
                cap.release()
                if not frames:
                    self.output_text.append("No frames were extracted from video.")
                    return
                # 将抽取的帧存储起来，传递给模型
                self.current_image = frames
                
                # 使用中间帧作为缩略图展示
                thumbnail_frame = frames[len(frames) // 2]
                qimage = QImage(
                    thumbnail_frame.tobytes(),
                    thumbnail_frame.width,
                    thumbnail_frame.height,
                    QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                
                self.default_question = "请详细描述这个视频及其动态变化"
            
            # 清空对话历史和输出框，重置输入区域
            self.chat_history = []
            self.output_text.clear()
            self.input_text.clear()
            self.input_text.setText(self.default_question)
            self.input_text.moveCursor(self.input_text.textCursor().End)
            self.send_btn.setEnabled(True)
            
    def processInput(self):
        if not self.ov_model:
            self.output_text.setText("Model is not ready yet. Please wait...")
            return
        
        if not self.current_image:
            self.output_text.setText("Please load an image first!")
            return
            
        # 获取用户输入的文本
        question = self.input_text.toPlainText().strip()
        self.chat_history.append(("user", question))
        self.updateChatDisplay()
        
        self.send_btn.setEnabled(False)
        self.input_text.clear()  # 清空输入框
        # 重新设置默认问题
        self.input_text.setText(self.default_question)
        self.input_text.moveCursor(self.input_text.textCursor().End)
        self.current_response = ""
        
        # 不再修改 question，而是直接使用原始文本
        # image 参数会自动处理单帧或多帧的情况
        self.inference_thread = InferenceThread(
            self.ov_model, 
            self.current_image,  # 可以是单张图片或图片列表
            question,            # 保持原始文本格式
            self.tokenizer
        )
        self.inference_thread.output_signal.connect(self.updateOutput)
        self.inference_thread.finished.connect(self.onInferenceComplete)
        self.inference_thread.start()
        
    def updateOutput(self, text):
        self.current_response += text
        self.updateChatDisplay()
        
    def onInferenceComplete(self):
        # 将完整的响应添加到对话历史
        self.chat_history.append(("assistant", self.current_response))
        self.updateChatDisplay()
        self.send_btn.setEnabled(True)
        
    def updateChatDisplay(self):
        display_text = ""
        for role, content in self.chat_history:
            if role == "user":
                display_text += f"#### {content}\n\n"
            else:
                display_text += f">>>> {content}\n\n"
                
        # 如果正在生成回复，添加当前的不完整回复
        if hasattr(self, 'current_response') and self.chat_history[-1][0] == "user":
            display_text += f">>>> {self.current_response}"
            
        self.output_text.setText(display_text)
        # 滚动到底部
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def handle_media_state_changed(self, state):
        # 可以用于处理播放状态的变化
        pass

    def handle_media_status_changed(self, status):
        # 当视频播放结束时
        if status == QMediaPlayer.EndOfMedia:
            # 将视频位置设置回开始
            self.media_player.setPosition(0)
            # 暂停播放
            self.media_player.pause()

def main():
    parser = argparse.ArgumentParser(description='Video Analysis GUI')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to extract from video (default: 16)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for image/video frames (0.1-10, default: 1.0)')
    args = parser.parse_args()
    
    # 验证scale参数范围
    if not 0.1 <= args.scale <= 10:
        parser.error("Scale factor must be between 0.1 and 10")
    
    app = QApplication(sys.argv)
    gui = VideoAnalyzeApp(num_frames=args.num_frames, scale=args.scale)
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()