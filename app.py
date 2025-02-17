import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QPushButton, QTextEdit, 
                            QLabel)
from PyQt5.QtCore import Qt

class PosterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poster Generator")
        self.setGeometry(100, 100, 1200, 800)
        
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
        left_layout = QVBoxLayout(left_widget)
        
        # 第一行控件组
        top_controls = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["模型1", "模型2", "模型3"])
        
        self.size_combo = QComboBox()
        self.size_combo.addItems(["大", "中", "小"])
        
        self.load_image_btn = QPushButton("加载原始图片")
        
        top_controls.addWidget(self.model_combo)
        top_controls.addWidget(self.size_combo)
        top_controls.addWidget(self.load_image_btn)
        
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
        right_layout = QVBoxLayout(right_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        self.image_label.setText("图片显示区域")
        right_layout.addWidget(self.image_label)
        
        # 将左右两侧添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        # 设置左右两侧比例为1:1
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PosterGUI()
    window.show()
    sys.exit(app.exec_())