from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
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
    def poster_generation_process(size_str: str, image_label: QLabel, image_processor: ImageProcessing) -> str:
        """生成并显示空白图片，如果有输入图片则分析其内容
        
        Args:
            size_str: str - 尺寸字符串 (例如 "1920x1080")
            image_label: QLabel - 用于显示图片的标签
            image_processor: ImageProcessing - 图片处理器实例
            
        Returns:
            str - 如果有输入图片，返回图片分析结果；否则返回空字符串
        """
        # 生成并缩放图片
        scaled_pixmap = ImageGenerator.generate_blank_poster(size_str, image_label.size())
        
        # 显示图片
        ImageGenerator.update_poster_image_preview(image_label, scaled_pixmap)
        
        # 检查是否有输入图片
        image_paths = image_processor.get_image_paths()
        if not image_paths:
            print("错误：没有找到输入图片")
            return ""
            
        try:
            # 尝试打开第一张图片
            input_image = Image.open(image_paths[0])
            return ImageGenerator.understand_input_image(input_image)
        except Exception as e:
            print(f"错误：无法打开图片 {image_paths[0]}: {str(e)}")
            return "" 