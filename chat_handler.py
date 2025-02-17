from PyQt5.QtCore import QObject, pyqtSignal
from llm_ollama import ChatWorker

class ChatHandler(QObject):
    """处理聊天相关的逻辑，作为GUI和LLM之间的中间层"""
    message_received = pyqtSignal(str)  # 用户消息信号
    ai_stream = pyqtSignal(str)         # AI流式响应信号
    ai_finished = pyqtSignal()          # AI响应完成信号
    error_occurred = pyqtSignal(str)    # 错误信号
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_response = ""
    
    def send_message(self, message: str, model: str):
        """发送消息到LLM"""
        if not message.strip():
            return
            
        self.message_received.emit(message)
        
        # 创建并启动worker
        self.worker = ChatWorker(message, model)
        self.worker.stream.connect(self.handle_stream)
        self.worker.finished.connect(self.handle_response)
        self.worker.error.connect(self.handle_error)
        self.worker.start()
    
    def handle_stream(self, text: str):
        """处理流式输出"""
        self.current_response = text
        self.ai_stream.emit(text)
    
    def handle_response(self, full_response: str):
        """处理完整响应"""
        self.ai_finished.emit()
    
    def handle_error(self, error_msg: str):
        """处理错误"""
        self.error_occurred.emit(error_msg) 