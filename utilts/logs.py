#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : logs.py
# @Project : Concept
# @Author  : wsw
# @Time    : 2023/10/23 14:59


import os
from functools import wraps
from time import perf_counter
import time
from loguru import logger

from datetime import datetime
import json


class Logging:
    """
    根据时间、文件大小切割日志
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    def __init__(self, log_dir='log', max_size=50, retention="2 months", project_level='INFO'):
        self.log_dir = log_dir
        self.max_size = max_size
        self.retention = retention
        self.project_level = project_level
        self.logger = self.configure_logger()

    def configure_logger(self):
        """

        Returns:

        """
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        shared_config = {
            "level": "INFO",
            "enqueue": True,
            "backtrace": True,
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        }
        # 添加按照日期和大小切割的文件 handler
        logger.remove()
        logger.add(
            sink=f"{self.log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation="1 day",
            retention=self.retention,
            **shared_config
        )
        self.record_log('-----------log init successful-----------')

        return logger

    def record_log(self, message='', level='INFO', *args, **kwargs):
        """
        0 debug
        1 info
        2 warning
        3 error
        """

        # print(f"{self.RED}ONE{self.RESET}")
        # print(f"{self.GREEN}TWO{self.RESET}")
        level = str(level)

        if level == '2':
            logger.warning(message, *args, **kwargs)

        elif level == '3':
            logger.error(message, *args, **kwargs)
            current_dateTime = datetime.now()
            print(f"{self.RED} {current_dateTime} error={message}")
            # print(message)
        elif level == '0':
            logger.debug(message, *args, **kwargs)

        else:
            logger.info(message, *args, **kwargs)

    def get_log_path(self, message: str) -> str:
        """
        根据等级返回日志路径
        Args:
            message:

        Returns:

        """
        log_level = message.record["level"].name.lower()
        log_file = f"{log_level}.log"
        log_path = os.path.join(self.log_dir, log_file)

        return log_path

    def __getattr__(self, level: str):
        return getattr(self.logger, level)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        return logger.exception(msg, *args, exc_info=True, **kwargs)

    def catch(self, msg="异常!", level=2, inputs=False, outputs=False, timing=True):
        """
         日志装饰器，记录函数的名称、参数、返回值、运行时间和异常信息
            Args:
                logger: 日志记录器对象

            Returns:
                装饰器函数

        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):

                if inputs:
                    self.record_log(f'-----------分割线-----------')
                    self.record_log(
                        f'调用 {func.__name__} args: {args}; kwargs:{json.dumps(kwargs, ensure_ascii=False)}')

                start = perf_counter()  # 开始时间
                try:
                    result = func(*args, **kwargs)

                    end = perf_counter()  # 结束时间
                    duration = end - start

                    if outputs:
                        self.record_log(f" ### {func.__name__} 返回结果：{result}")
                    if timing:
                        self.record_log(f" ### {func.__name__}, 耗时：{duration:4f}s")
                    return result
                except Exception as e:
                    self.exception(f"{func.__name__}: {msg}")
                    self.info(f"---------异常分割线-----------")
                    # raise e

            return wrapper

        return decorator


logs = Logging()
