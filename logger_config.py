import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# 定义TRACE级别（比DEBUG更详细）
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def setup_logger(name, log_file=None, level=logging.INFO):
    """设置并返回具有指定名称和级别的日志记录器
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志记录器的级别
        
    返回:
        配置好的日志记录器实例
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除任何已有的处理器，避免重复配置
    if logger.handlers:
        logger.handlers = []
    
    # 创建控制台处理器，只显示INFO以上级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，创建文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 配置文件处理器，记录DEBUG及以上级别，限制大小为5MB，最多保留3个备份
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(logging.DEBUG)  # 文件中记录所有级别
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger

# 添加trace方法到Logger类
def trace(self, message, *args, **kwargs):
    """
    记录TRACE级别的日志消息
    """
    if self.isEnabledFor(TRACE):
        self.log(TRACE, message, *args, **kwargs)

# 将trace方法添加到Logger类
logging.Logger.trace = trace

# 全局默认日志器
def get_default_logger():
    """获取默认的全局日志记录器"""
    return setup_logger('default_logger', 'logs/application.log') 