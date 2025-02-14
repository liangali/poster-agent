from llama_index.llms.ollama import Ollama
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QComboBox, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import sys

model_list = ['qwen2.5:7b', 'qwen2.5:14b', 'deepseek-r1:7b', 'deepseek-r1:14b', 'deepseek-r1:7b-qwen-distill-q8_0']
use_model = model_list[3]

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatWorker(QThread):
    """处理AI对话的工作线程，支持流式输出"""
    finished = pyqtSignal(str)  # 完成信号，发送最终完整结果
    error = pyqtSignal(str)     # 错误信号
    stream = pyqtSignal(str)    # 流式输出信号

    def __init__(self, question, model):
        super().__init__()
        self.question = question
        self.model = model

    def run(self):
        try:
            llm = Ollama(
                model=self.model,
                temperature=0.7,
                timeout=120
            )
            response = llm.stream_complete(self.question)
            full_response = ""
            for chunk in response:
                if chunk and chunk.text:
                    # 如果返回的chunk.text是累计的，则每次直接使用最新的输出
                    self.stream.emit(chunk.text)
                    full_response = chunk.text
            self.finished.emit(full_response)
        except Exception as e:
            self.error.emit(str(e))

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 对话助手")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建中心部件和总体布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 第一行：模型选择控件
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型：")
        self.model_selector = QComboBox()
        self.model_selector.addItems(model_list)
        self.model_selector.setCurrentText(use_model)
        self.model_selector.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()  # 保持靠左对齐
        
        # 输入区域
        self.input_label = QLabel("请输入您的问题：")
        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(100)
        
        # 发送按钮（独占一行）
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_question)
        
        # 聊天记录输出区域
        self.output_label = QLabel("对话历史：")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        
        # 添加控件到总体布局
        layout.addLayout(model_layout)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_text)
        layout.addWidget(self.send_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_text)
        
        self.worker = None
        self.current_response = ""  # 当前AI响应内容

    def change_model(self, model_name):
        global use_model
        use_model = model_name
        self.output_text.append(f"\n系统：已切换到模型 {model_name}\n")

    def send_question(self):
        question = self.input_text.toPlainText().strip()
        if not question:
            return

        self.send_button.setEnabled(False)
        
        # 打印用户问题和预留AI回答区域
        self.output_text.append("\n【用户】")
        self.output_text.append(question)
        self.output_text.append("\n【AI助手】\n")
        
        # 清除输入框和当前响应缓存
        self.input_text.clear()
        self.current_response = ""
        
        # 获取当前下拉框选中的模型，并创建后台线程
        current_model = self.model_selector.currentText()
        self.worker = ChatWorker(question, current_model)
        self.worker.stream.connect(self.handle_stream)
        self.worker.finished.connect(self.handle_response)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def handle_stream(self, text):
        """
        处理流式输出：
        若返回的 text 是累计文本，则直接替换当前AI回答部分
        """
        # 直接使用流输出的最新文本，避免重复累计
        self.current_response = text

        current_text = self.output_text.toPlainText()
        try:
            last_ai_index = current_text.rindex("【AI助手】")
            base_text = current_text[:last_ai_index + len("【AI助手】\n")]
        except ValueError:
            base_text = current_text
        self.output_text.setPlainText(base_text + self.current_response)
        
        # 保持滚动到最下方
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def handle_response(self, full_response):
        """接收完整响应后重新启用发送按钮，并打印分割线"""
        self.send_button.setEnabled(True)
        self.output_text.append("\n" + "="*50 + "\n")

    def handle_error(self, error_msg):
        """处理错误，显示错误信息并恢复发送按钮"""
        current_text = self.output_text.toPlainText()
        try:
            last_ai_index = current_text.rindex("【AI助手】")
            base_text = current_text[:last_ai_index + len("【AI助手】\n")]
        except ValueError:
            base_text = current_text
        self.output_text.setPlainText(base_text)
        self.output_text.append(f"错误: {error_msg}")
        self.send_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    # 设置全局字体为微软雅黑，字号可根据需要自行调整
    app.setFont(QFont("Microsoft YaHei", 8))
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 