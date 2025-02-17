import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QPushButton, QTextEdit, 
                            QLabel, QSizePolicy)
from PyQt5.QtCore import Qt
from llm_ollama import MODEL_LIST
from chat_handler import ChatHandler
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from generate import ImageGenerator

class PosterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poster Generator")
        self.setGeometry(100, 100, 1800, 1000)
        
        # 设置全局字体
        self.setStyleSheet("""
            * {
                font-family: "Microsoft YaHei";
            }
            QLabel {
                background-color: #f0f0f0;
                font-family: "Microsoft YaHei";
            }
        """)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(main_widget)
        
        # 创建左侧部件和布局
        left_widget = QWidget()
        left_widget.setStyleSheet("""
            QWidget#leftContainer {
                border: 1px solid #808080;
            }
        """)
        left_widget.setObjectName("leftContainer")  # 设置对象名称
        left_layout = QVBoxLayout(left_widget)
        
        # 第一行控件组
        top_controls = QHBoxLayout()
        
        # 创建LLM模型组合控件的容器
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)  # 移除内边距
        model_layout.setSpacing(0)  # 移除间距
        model_label = QLabel("LLM模型: ")
        model_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 允许水平扩展
        self.model_combo.addItems(MODEL_LIST)
        # 设置默认模型为'deepseek-r1:7b'
        default_model = MODEL_LIST[2]
        default_index = MODEL_LIST.index(default_model)
        self.model_combo.setCurrentIndex(default_index)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # 创建图片大小组合控件的容器
        size_container = QWidget()
        size_layout = QHBoxLayout(size_container)
        size_layout.setContentsMargins(0, 0, 0, 0)  # 移除内边距
        size_layout.setSpacing(0)  # 移除间距
        size_label = QLabel("海报大小: ")
        size_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
        self.size_combo = QComboBox()
        self.size_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 允许水平扩展
        self.size_combo.addItems(["1080x1920", "1920x1080", "1200x1200"])
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_combo)
        
        self.load_image_btn = QPushButton("加载图片")
        self.generate_btn = QPushButton("开始生成")  # 新增按钮
        
        # 添加所有控件到顶部布局并设置新的宽度比例
        top_controls.addWidget(model_container, 3)        # 从4改为3
        top_controls.addWidget(size_container, 3)         # 从4改为3
        top_controls.addWidget(self.load_image_btn, 2)    # 保持2
        top_controls.addWidget(self.generate_btn, 2)      # 新增，比例为2
        
        # 第二行：LLM输出框
        self.llm_output = QTextEdit()
        self.llm_output.setReadOnly(True)
        self.llm_output.setPlaceholderText("LLM输出将显示在这里...")
        
        # 第三行控件组
        bottom_controls = QHBoxLayout()
        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(100)
        self.user_input.setPlaceholderText("在这里输入你的提示...")
        
        self.send_btn = QPushButton("发送")
        self.clear_btn = QPushButton("清除")
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.send_btn)
        button_layout.addWidget(self.clear_btn)
        
        bottom_controls.addWidget(self.user_input)
        bottom_controls.addLayout(button_layout)
        
        # 添加所有控件到左侧布局
        left_layout.addLayout(top_controls)
        left_layout.addWidget(self.llm_output)
        left_layout.addLayout(bottom_controls)
        
        # 创建右侧图片显示区域
        right_widget = QWidget()
        right_widget.setStyleSheet("""
            QWidget#rightContainer {
                border: 1px solid #808080;
            }
        """)
        right_widget.setObjectName("rightContainer")  # 设置对象名称
        right_layout = QVBoxLayout(right_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("图片显示区域")
        right_layout.addWidget(self.image_label)
        
        # 将左右两侧添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        # 设置左右两侧比例为1:1
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        
        # 初始化聊天处理器
        self.chat_handler = ChatHandler()
        self.setup_chat_connections()
        
        # 连接开始生成按钮的点击事件
        self.generate_btn.clicked.connect(self.generate_poster)
        
    def setup_chat_connections(self):
        """设置聊天相关的信号连接"""
        # 连接发送和清除按钮
        self.send_btn.clicked.connect(self.send_message)
        self.clear_btn.clicked.connect(self.clear_output)
        
        # 连接聊天处理器的信号
        self.chat_handler.message_received.connect(self.display_user_message)
        self.chat_handler.ai_stream.connect(self.update_ai_response)
        self.chat_handler.ai_finished.connect(self.on_response_finished)
        self.chat_handler.error_occurred.connect(self.handle_error)
        
    def send_message(self):
        """处理发送消息"""
        question = self.user_input.toPlainText().strip()
        if not question:
            return
            
        self.send_btn.setEnabled(False)
        self.user_input.clear()
        
        # 发送消息到聊天处理器
        current_model = self.model_combo.currentText()
        self.chat_handler.send_message(question, current_model)
        
    def display_user_message(self, message: str):
        """显示用户消息"""
        self.llm_output.append("\n【用户】")
        self.llm_output.append(message)
        self.llm_output.append("\n【AI助手】\n")
        
    def update_ai_response(self, text: str):
        """更新AI响应"""
        current_text = self.llm_output.toPlainText()
        try:
            last_ai_index = current_text.rindex("【AI助手】")
            base_text = current_text[:last_ai_index + len("【AI助手】\n")]
        except ValueError:
            base_text = current_text
            
        self.llm_output.setPlainText(base_text + text)
        self.llm_output.verticalScrollBar().setValue(
            self.llm_output.verticalScrollBar().maximum()
        )
        
    def on_response_finished(self):
        """AI响应完成的处理"""
        self.send_btn.setEnabled(True)
        self.llm_output.append("\n" + "="*50 + "\n")
        
    def handle_error(self, error_msg: str):
        """处理错误"""
        current_text = self.llm_output.toPlainText()
        try:
            last_ai_index = current_text.rindex("【AI助手】")
            base_text = current_text[:last_ai_index + len("【AI助手】\n")]
        except ValueError:
            base_text = current_text
        self.llm_output.setPlainText(base_text)
        self.llm_output.append(f"错误: {error_msg}")
        self.send_btn.setEnabled(True)
        
    def clear_output(self):
        """清除输出"""
        self.llm_output.clear()

    def generate_poster(self):
        """生成并显示空白图片"""
        size_str = self.size_combo.currentText()
        ImageGenerator.poster_generation_process(size_str, self.image_label)
        
    def resizeEvent(self, event):
        """窗口大小改变时重新调整图片大小"""
        super().resizeEvent(event)
        if self.image_label.pixmap() is not None:
            scaled_pixmap = ImageGenerator.scale_pixmap(
                self.image_label.pixmap(),
                self.image_label.size()
            )
            self.image_label.setPixmap(scaled_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PosterGUI()
    window.show()
    sys.exit(app.exec_())