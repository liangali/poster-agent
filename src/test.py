from PIL import Image
from analyze import AnalyzeImage

# 初始化分析器
analyzer = AnalyzeImage(model_dir='../../models/minicpm_v_2_6', device='GPU')

# 加载图片
image = Image.open('C:\\data\\code\\poster_agent_code\\test\\img1.jpg')

# 方式 1：使用 analyze 方法
response = analyzer.analyze(image, "请描述这张图片的内容")
print(response)

# 方式 2：直接调用实例
response = analyzer(image, "这张图片里有什么？")
print(response)

# 使用流式输出
for token in analyzer.analyze(image, "描述图片", stream=True):
    print(token, end='', flush=True)