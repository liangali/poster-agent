import os
import json
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

class LLMLogger:
    def __init__(self):
        # 创建上两级目录下的log文件夹
        self.log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'log'))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建新的日志文件，使用时间戳命名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'llm_log_{timestamp}.log')
        
        # 配置日志记录器
        self.logger = logging.getLogger('LLMLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        self.logger.handlers = []
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式 - 移除日志级别，保持更简洁的输出
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.logger.addHandler(file_handler)
        
        self.logger.info("=== LLM Logger Initialized ===")
        self.conversation_counter = 0

    def _format_json(self, obj: Any) -> str:
        """格式化 JSON 对象为易读的字符串"""
        try:
            if isinstance(obj, str):
                try:
                    # 尝试解析JSON字符串
                    obj = json.loads(obj)
                except json.JSONDecodeError:
                    # 如果不是JSON字符串，直接返回
                    return obj
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return str(obj)

    def _format_content(self, content: Any) -> str:
        """格式化消息内容"""
        if isinstance(content, list):
            return "\n".join(self._format_json(item) for item in content)
        elif isinstance(content, dict):
            return self._format_json(content)
        else:
            return str(content)

    def log_conversation_start(self) -> None:
        """记录新会话的开始"""
        self.conversation_counter += 1
        self.logger.info("\n==================================================")
        self.logger.info(f"Conversation #{self.conversation_counter} Started")
        self.logger.info("\n==================================================")

    def log_messages(self, messages: List[Dict[str, str]]) -> None:
        """记录输入消息"""
        self.log_conversation_start()
        self.logger.info("=== LLM Input Messages ===")
        for msg in messages:
            self.logger.info(f"Role: {msg.get('role')}")
            content = msg.get('content')
            self.logger.info(f"Content:\n{self._format_content(content)}")
            
            if 'tool_calls' in msg and msg['tool_calls']:
                self.logger.info(f"Tool Calls:\n{self._format_json(msg['tool_calls'])}")
            if 'tool_call_id' in msg:
                self.logger.info(f"Tool Call ID: {msg['tool_call_id']}")
            self.logger.info("---")
    
    def log_tools(self, tools: Optional[List[Any]]) -> None:
        """记录可用工具信息"""
        if not tools:
            return
        
        self.logger.info("\n=== Available Tools ===")
        for tool in tools:
            try:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputs": tool.inputs
                }
                self.logger.info(self._format_json(tool_info))
            except AttributeError:
                self.logger.info(str(tool))
        self.logger.info("---")
    
    def log_response(self, response: Any) -> None:
        """记录LLM响应"""
        self.logger.info("\n=== LLM Response ===")
        
        # 尝试提取和格式化响应信息
        try:
            if hasattr(response, 'content'):
                # 处理 ChatMessage 对象
                content = {
                    'role': getattr(response, 'role', 'assistant'),
                    'content': response.content
                }
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    content['tool_calls'] = response.tool_calls
                self.logger.info(self._format_json(content))
            else:
                # 处理其他类型的响应
                self.logger.info(self._format_json(response))
        except Exception as e:
            self.logger.warning(f"Failed to format response: {str(e)}")
            self.logger.info(f"Raw response: {str(response)}")
        
        self.logger.info("---")
    
    def log_error(self, error: Exception) -> None:
        """记录错误信息"""
        self.logger.error("\n=== Error in LLM Call ===")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        self.logger.error("---\n") 