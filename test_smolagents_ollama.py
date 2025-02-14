from smolagents import ToolCallingAgent, LiteLLMModel, tool  
from ollama import Client  
import os  
from dotenv import load_dotenv  
  
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
  
    def generate(self, messages, tools=None):  
        response = client.chat(  
            model=use_model,  
            messages=messages,  
            options={"temperature": 0.7}  
        )  
        return response['message']['content']  
  
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