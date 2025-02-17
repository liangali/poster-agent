from smolagents import ToolCallingAgent, LiteLLMModel, tool, Tool, ChatMessage  
from ollama import Client  
import os  
from dotenv import load_dotenv  
import json
from utils.llm_logger import LLMLogger
from typing import List, Dict, Optional, Any
  
model_list = ['qwen2.5:7b', 'qwen2.5:14b', 'deepseek-r1:7b', 'deepseek-r1:14b', 'deepseek-r1:7b-qwen-distill-q8_0']
use_model = model_list[1]

# 加载本地环境变量  
load_dotenv()  
  
# 配置Ollama客户端  
client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))  
  
# 定义工具函数  
@tool  
def get_weather(location: str) -> str:  
    """获取指定城市的实时天气信息（模拟数据）。

    Args:
        location: 要查询天气的城市名称。需要输入有效的中国城市名称。

    Returns:
        str: 包含温度、天气状况、风向、湿度等信息的天气报告。
    """  
    return f"{location}的当前气温为23℃，晴，东南风2级，空气湿度65%。"  
  
@tool
def get_product_description(product_name: str) -> Dict[str, str]:
    """获取产品的详细描述信息。

    Args:
        product_name: 产品名称。

    Returns:
        Dict[str, str]: 包含产品详细信息的字典。
    """
    return {
        "name": "极光保温杯",
        "capacity": "500ml",
        "material": "304不锈钢",
        "features": "双层真空保温，12小时保温效果，简约北欧风设计",
        "price": "¥139",
        "colors": ["极光银", "玫瑰金", "星空黑"]
    }

@tool
def plan_poster_layout(product_info: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """规划海报布局。

    Args:
        product_info: 产品信息字典。

    Returns:
        Dict[str, Dict[str, str]]: 海报各元素的布局信息。
    """
    return {
        "product_image": {
            "position": "居中偏右",
            "size": "占据画面50%",
            "style": "45度倾斜展示"
        },
        "background": {
            "position": "全画面",
            "style": "渐变背景",
            "color_scheme": "根据产品颜色搭配冷暖色调"
        },
        "text_elements": {
            "title": "左上角20%区域",
            "features": "左侧30%区域",
            "price": "右下角15%区域",
            "slogan": "底部居中位置"
        }
    }

@tool
def get_product_image(product_name: str) -> str:
    """获取产品主图的抠图。

    Args:
        product_name: 产品名称。

    Returns:
        str: 模拟的产品图片描述。
    """
    return "[模拟输出] 已生成保温杯透明背景主图，45度展示角度，突出杯身质感和材质"

@tool
def get_background_image(style_description: str) -> str:
    """获取海报背景图。

    Args:
        style_description: 背景风格描述。

    Returns:
        str: 模拟的背景图片描述。
    """
    return "[模拟输出] 已生成渐变背景，从左上角的浅蓝色渐变到右下角的深紫色，营造高级感"

@tool
def generate_poster_text(product_info: Dict[str, str]) -> Dict[str, str]:
    """生成海报文案。

    Args:
        product_info: 产品信息字典。

    Returns:
        Dict[str, str]: 包含各个文案元素的字典。
    """
    return {
        "title": "极光保温杯 - 让饮品保持最佳温度",
        "features": "• 12小时持久保温\n• 304食品级不锈钢\n• 北欧简约设计\n• 500ml大容量",
        "price": "限时特惠 ¥139",
        "slogan": "品质生活，从一杯开始"
    }

@tool
def compose_final_poster(
    product_image: str,
    background: str,
    text_content: Dict[str, str],
    layout: Dict[str, Dict[str, str]]
) -> str:
    """将所有元素组合成最终海报。

    Args:
        product_image: 产品图片。
        background: 背景图片。
        text_content: 文案内容。
        layout: 布局信息。

    Returns:
        str: 模拟的最终海报描述。
    """
    return f"""[模拟输出] 已生成最终海报：
    1. 背景：{background}
    2. 产品主图：{product_image}
    3. 文案布局：
       - 标题：{text_content['title']} ({layout['text_elements']['title']})
       - 特性：{text_content['features']} ({layout['text_elements']['features']})
       - 价格：{text_content['price']} ({layout['text_elements']['price']})
       - 标语：{text_content['slogan']} ({layout['text_elements']['slogan']})
    """

# 创建本地LLM模型实例  
class OllamaModel(LiteLLMModel):  
    def __init__(self):  
        super().__init__(model_id=f"ollama/{use_model}")
        self.logger = LLMLogger()

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:  
        # 记录日志
        self.logger.log_messages(messages)

        # 直接调用父类的 __call__ 方法，smolagents 会处理工具转换
        response = super().__call__(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,  # 直接传递工具列表
            **kwargs
        )
        
        if isinstance(response, dict):
            self.logger.log_response(response)
        else:
            self.logger.log_response({"content": str(response)})
        
        return response
  
# 初始化智能体  
agent = ToolCallingAgent(  
    tools=[
        get_product_description,
        plan_poster_layout,
        get_product_image,
        get_background_image,
        generate_poster_text,
        compose_final_poster
    ],  
    model=OllamaModel()  
)  
  
# 执行示例对话  
if __name__ == "__main__":  
    print("系统已启动，输入'exit'退出")  
    while True:  
        query = input("\n用户提问: ")  
        if query.lower() == 'exit':  
            break  
          
        response = agent.run(query)  
        print(f"\nAI回复: {response}")  