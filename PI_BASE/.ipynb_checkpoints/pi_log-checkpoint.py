#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI项目专用日志模块
提供统一的日志记录功能，支持不同级别的日志输出
"""

import logging
import os
from datetime import datetime
from typing import Optional
import sys

# 创建全局logger实例
_logger = None

def setup_file_logging(log_file_path: str = "./log/ahum.log", level: int = logging.INFO):
    """
    设置文件日志记录
    
    Args:
        log_file_path: 日志文件路径
        level: 日志级别
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    return file_handler

def get_logger(name: str = "PI_LLM", level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    """
    获取或创建logger实例
    
    Args:
        name: logger名称
        level: 日志级别
        log_to_file: 是否记录到文件
        
    Returns:
        logging.Logger: 配置好的logger实例
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # 创建logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    # 避免重复添加handler
    if _logger.handlers:
        return _logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    _logger.addHandler(console_handler)
    
    # 添加文件处理器（如果需要）
    if log_to_file:
        try:
            file_handler = setup_file_logging()
            _logger.addHandler(file_handler)
            log_info(f"📝 日志文件已启用: ./log/ahum.log")
        except Exception as e:
            log_warning(f"无法设置文件日志记录: {e}")
    
    return _logger

def log_info(message: str, logger: Optional[logging.Logger] = None):
    """记录INFO级别日志"""
    if logger is None:
        logger = get_logger()
    logger.info(message)

def log_debug(message: str, logger: Optional[logging.Logger] = None):
    """记录DEBUG级别日志"""
    if logger is None:
        logger = get_logger()
    logger.debug(message)

def log_warning(message: str, logger: Optional[logging.Logger] = None):
    """记录WARNING级别日志"""
    if logger is None:
        logger = get_logger()
    logger.warning(message)

def log_error(message: str, logger: Optional[logging.Logger] = None):
    """记录ERROR级别日志"""
    if logger is None:
        logger = get_logger()
    logger.error(message)

def log_critical(message: str, logger: Optional[logging.Logger] = None):
    """记录CRITICAL级别日志"""
    if logger is None:
        logger = get_logger()
    logger.critical(message)

def log_model_loading_start(model_path: str, logger: Optional[logging.Logger] = None):
    """记录模型加载开始"""
    log_info(f"🚀 开始加载模型: {model_path}", logger)

def log_model_loading_success(model_path: str, device: str, logger: Optional[logging.Logger] = None):
    """记录模型加载成功"""
    log_info(f"✅ 模型加载成功: {model_path} (设备: {device})", logger)

def log_model_loading_failed(model_path: str, error: str, logger: Optional[logging.Logger] = None):
    """记录模型加载失败"""
    log_error(f"❌ 模型加载失败: {model_path} - 错误: {error}", logger)

def log_cache_hit(model_path: str, logger: Optional[logging.Logger] = None):
    """记录缓存命中"""
    log_info(f"🔄 缓存命中，使用已加载模型: {model_path}", logger)

def log_cache_miss(model_path: str, logger: Optional[logging.Logger] = None):
    """记录缓存未命中"""
    log_info(f"🔍 缓存未命中，开始加载新模型: {model_path}", logger)

def log_cache_store(model_path: str, logger: Optional[logging.Logger] = None):
    """记录模型缓存存储"""
    log_info(f"💾 模型已缓存: {model_path}", logger)

def log_cache_clear(model_path: str = "ALL", logger: Optional[logging.Logger] = None):
    """记录缓存清理"""
    if model_path == "ALL":
        log_info("🧹 已清理所有模型缓存", logger)
    else:
        log_info(f"🧹 已清理模型缓存: {model_path}", logger)

def log_device_detection(device: str, dtype: str, logger: Optional[logging.Logger] = None):
    """记录设备检测结果"""
    log_info(f"🖥️ 检测到设备: {device}, 数据类型: {dtype}", logger)

def log_model_test_start(logger: Optional[logging.Logger] = None):
    """记录模型测试开始"""
    log_info("🧪 开始模型功能测试...", logger)

def log_model_test_complete(logger: Optional[logging.Logger] = None):
    """记录模型测试完成"""
    log_info("✅ 模型功能测试完成", logger)

def log_download_start(model_id: str, model_dir: str, logger: Optional[logging.Logger] = None):
    """记录模型下载开始"""
    log_info(f"📥 开始下载模型: {model_id} 到 {model_dir}", logger)

def log_download_complete(local_dir: str, logger: Optional[logging.Logger] = None):
    """记录模型下载完成"""
    log_info(f"✅ 模型下载完成，保存位置: {local_dir}", logger)

def log_file_exists(model_path: str, logger: Optional[logging.Logger] = None):
    """记录文件已存在"""
    log_info(f"📁 模型文件已存在: {model_path}", logger)

# 便捷函数别名
info = log_info
debug = log_debug
warning = log_warning
error = log_error
critical = log_critical

if __name__ == "__main__":
    # 测试日志功能
    logger = get_logger("TEST")
    
    print("=== PI日志模块测试 ===")
    log_info("这是INFO级别日志")
    log_debug("这是DEBUG级别日志")
    log_warning("这是WARNING级别日志")
    log_error("这是ERROR级别日志")
    log_critical("这是CRITICAL级别日志")
    
    log_model_loading_start("/path/to/model")
    log_model_loading_success("/path/to/model", "mps")
    log_cache_hit("/path/to/model")
    log_cache_store("/path/to/model")