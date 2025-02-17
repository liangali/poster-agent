from llama_index.llms.ollama import Ollama
from PyQt5.QtCore import QThread, pyqtSignal
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可用的模型列表
MODEL_LIST = ['qwen2.5:7b', 'qwen2.5:14b', 'deepseek-r1:7b', 'deepseek-r1:14b']

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
                    self.stream.emit(chunk.text)
                    full_response = chunk.text
            self.finished.emit(full_response)
        except Exception as e:
            logger.error(f"LLM处理错误: {str(e)}")
            self.error.emit(str(e)) 