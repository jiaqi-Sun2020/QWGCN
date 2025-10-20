import os
import sys
from datetime import datetime


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout  # 保留原始输出流
        self.log = open(log_path, "a", encoding="utf-8")  # 追加写入日志文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(log_dir='logs', log_filename=None):
    """
    设置日志系统，将所有print内容同时输出到控制台和txt文件中。

    参数:
        log_dir (str): 日志保存文件夹。
        log_filename (str or None): 日志文件名（默认使用当前时间）。
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"

    log_path = os.path.join(log_dir, log_filename)
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout  # 捕捉错误输出

    print(f"[日志已启动] 所有输出将保存至: {log_path}")
