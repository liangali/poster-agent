from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QTextEdit, QApplication
import numpy as np
from image import ImageProcessing
from analyze import AnalyzeImage

class ImageGenerator:
    """处理图片生成和转换的类"""
    
    @staticmethod
    def create_blank_image(width: int, height: int) -> Image.Image:
        """创建空白图片"""
        return Image.new('RGB', (width, height), 'white')
    
    @staticmethod
    def pil_to_pixmap(pil_image: Image.Image) -> QPixmap:
        """将PIL图片转换为QPixmap"""
        # 将PIL图片转换为numpy数组
        img_array = np.array(pil_image)
        height, width, channels = img_array.shape
        
        # 转换为QImage
        bytes_per_line = channels * width
        q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 转换为QPixmap
        return QPixmap.fromImage(q_img)
    
    @staticmethod
    def scale_pixmap(pixmap: QPixmap, target_size, keep_aspect=True) -> QPixmap:
        """缩放QPixmap到指定大小"""
        if keep_aspect:
            return pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        return pixmap.scaled(
            target_size,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
    
    @staticmethod
    def generate_blank_poster(size_str: str, target_size) -> QPixmap:
        """生成并缩放图片"""
        # 解析尺寸字符串
        width, height = map(int, size_str.split('x'))
        
        # 创建空白图片
        blank_image = ImageGenerator.create_blank_image(width, height)
        
        # 转换为QPixmap
        pixmap = ImageGenerator.pil_to_pixmap(blank_image)
        
        # 缩放到合适大小
        return ImageGenerator.scale_pixmap(pixmap, target_size)
    
    @staticmethod
    def update_poster_image_preview(image_label: QLabel, pixmap: QPixmap):
        """更新图片预览"""
        image_label.setPixmap(pixmap)
    
    @staticmethod
    def understand_input_image(image: Image.Image) -> str:
        """分析输入图片的内容
        
        Args:
            image: PIL.Image 对象
            
        Returns:
            str: 图片内容的描述
        """
        # 初始化分析器
        analyzer = AnalyzeImage(model_dir='../../models/minicpm_v_2_6', device='GPU')
        
        # 分析图片内容
        description = analyzer.analyze(image, "请详细描述这张图片的内容")
        
        return description

    @staticmethod
    def append_to_output(text_widget: QTextEdit, message: str):
        """向输出文本框添加消息并立即显示
        
        Args:
            text_widget: QTextEdit - 输出文本框
            message: str - 要显示的消息
        """
        text_widget.append(message)
        # 强制更新UI
        QApplication.processEvents()
        # 滚动到底部
        text_widget.verticalScrollBar().setValue(
            text_widget.verticalScrollBar().maximum()
        )

    @staticmethod
    def poster_generation_process(size_str: str, image_label: QLabel, image_processor: ImageProcessing) -> str:
        """生成并显示空白图片，如果有输入图片则分析其内容"""
        # 获取输出文本框引用
        output_text = image_label.window().findChild(QTextEdit)
        
        # 生成并缩放图片
        ImageGenerator.append_to_output(output_text, f"正在生成 {size_str} 尺寸的空白图片...")
        scaled_pixmap = ImageGenerator.generate_blank_poster(size_str, image_label.size())
        
        # 显示图片
        ImageGenerator.append_to_output(output_text, "正在更新图片预览...")
        ImageGenerator.update_poster_image_preview(image_label, scaled_pixmap)
        
        # 检查是否有输入图片
        ImageGenerator.append_to_output(output_text, "正在检查输入图片...")
        image_paths = image_processor.get_image_paths()
        if not image_paths:
            ImageGenerator.append_to_output(output_text, "错误：没有找到输入图片")
            return ""
            
        try:
            # 尝试打开第一张图片
            ImageGenerator.append_to_output(output_text, f"正在打开图片：{image_paths[0]}")
            input_image = Image.open(image_paths[0])
            
            ImageGenerator.append_to_output(output_text, "正在分析图片内容，请稍候...")
            result = ImageGenerator.understand_input_image(input_image)
            
            # 显示分析结果
            ImageGenerator.append_to_output(output_text, "\n分析结果：")
            ImageGenerator.append_to_output(output_text, result)
            
            return result
            
        except Exception as e:
            error_msg = f"错误：无法打开图片 {image_paths[0]}: {str(e)}"
            ImageGenerator.append_to_output(output_text, error_msg)
            return ""