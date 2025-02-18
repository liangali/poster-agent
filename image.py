import numpy as np
from PIL import Image

class ImageProcessing:
    def __init__(self):
        self.image_paths = []  # 存储图片路径
        self.rgb_images = []   # 存储RGB格式的图片数据
        
    def load_images(self, file_paths):
        """加载并处理图片
        
        Args:
            file_paths (list): 图片文件路径列表
        """
        self.image_paths = file_paths
        self.rgb_images = []
        
        for path in self.image_paths:
            try:
                # 使用PIL打开图片
                with Image.open(path) as img:
                    # 转换为RGB模式
                    rgb_img = img.convert('RGB')
                    # 转换为numpy数组
                    img_array = np.array(rgb_img)
                    self.rgb_images.append(img_array)
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                
    def get_image_count(self):
        """返回已加载的图片数量"""
        return len(self.rgb_images)
    
    def get_image_paths(self):
        """返回图片路径列表"""
        return self.image_paths
    
    def get_rgb_images(self):
        """返回RGB图片数据列表"""
        return self.rgb_images 