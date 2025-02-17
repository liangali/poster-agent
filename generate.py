from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
import numpy as np

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
    def poster_generation_process(size_str: str, image_label: QLabel):
        """生成并显示空白图片"""
        # 生成并缩放图片
        scaled_pixmap = ImageGenerator.generate_blank_poster(size_str, image_label.size())
        
        # 显示图片
        ImageGenerator.update_poster_image_preview(image_label, scaled_pixmap) 