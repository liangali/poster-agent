from smolagents import ToolCallingAgent, LiteLLMModel, tool, Tool, ChatMessage  
from ollama import Client  
import os  
from dotenv import load_dotenv  
import json
from utils.llm_logger import LLMLogger
from typing import List, Dict, Optional, Any
  
model_list = ['qwen2.5:7b', 'qwen2.5:14b', 'deepseek-r1:7b', 'deepseek-r1:14b', 'deepseek-r1:7b-qwen-distill-q8_0']
use_model = model_list[0]

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
    tools=[get_weather],  
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