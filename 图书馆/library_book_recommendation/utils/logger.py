import logging
import os
import time


class Logger:
    def __init__(self, log_file):
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, log_file)
        self.logger = self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def get_logger(self):
        return self.logger


def main():
    """测试Logger类的核心功能：目录创建、文件生成、日志输出、重复处理器避免"""
    # 测试日志文件名（包含时间戳，避免重复测试冲突）
    test_log_file = f"test_log_{int(time.time())}.log"
    print(f"开始测试Logger类，测试日志文件：{test_log_file}\n")

    # 1. 初始化Logger实例，验证日志目录是否创建
    logger_instance = Logger(test_log_file)
    log_dir_exists = os.path.exists(logger_instance.log_dir)
    print(f"【测试1：日志目录创建】{'通过' if log_dir_exists else '失败'} - 目录路径：{logger_instance.log_dir}")

    # 2. 获取logger，记录不同级别日志，验证输出
    logger = logger_instance.get_logger()
    test_messages = [
        (logging.INFO, "这是一条INFO级别的测试日志"),
        (logging.WARNING, "这是一条WARNING级别的测试日志"),
        (logging.ERROR, "这是一条ERROR级别的测试日志")
    ]
    print("\n【测试2：日志输出验证】")
    for level, msg in test_messages:
        logger.log(level, msg)  # 触发日志输出（控制台可见）

    # 3. 验证日志文件是否生成且内容正确
    log_file_exists = os.path.exists(logger_instance.log_path)
    print(f"\n【测试3：日志文件生成】{'通过' if log_file_exists else '失败'} - 文件路径：{logger_instance.log_path}")

    if log_file_exists:
        # 读取日志文件内容，验证格式和消息
        with open(logger_instance.log_path, 'r', encoding='utf-8') as f:
            log_content = f.readlines()

        # 检查日志条目数量是否匹配（每条测试消息对应一条日志）
        content_check = len(log_content) == len(test_messages)
        print(
            f"【测试4：日志内容完整性】{'通过' if content_check else '失败'} - 预期{len(test_messages)}条，实际{len(log_content)}条")

        # 检查每条日志格式是否正确（文件日志格式：时间 - 级别 - 消息）
        format_check = True
        for i, line in enumerate(log_content):
            expected_level = logging.getLevelName(test_messages[i][0])
            expected_msg = test_messages[i][1]
            # 验证级别和消息是否包含在日志中
            if expected_level not in line or expected_msg not in line:
                format_check = False
                print(f"  日志格式错误：第{i + 1}行 - {line.strip()}")
        print(f"【测试5：日志格式正确性】{'通过' if format_check else '失败'}")

    # 4. 验证重复创建Logger是否会导致重复handler（重复输出）
    print("\n【测试6：重复处理器避免】")
    logger_instance2 = Logger(test_log_file)  # 同一日志文件创建第二个实例
    logger2 = logger_instance2.get_logger()
    logger2.info("测试重复handler：这条日志应只输出一次")  # 若控制台只输出一次，说明通过

    # 检查日志文件中是否只记录一次（避免重复）
    if log_file_exists:
        with open(logger_instance.log_path, 'r', encoding='utf-8') as f:
            new_content = f.readlines()
        duplicate_check = len(new_content) == len(test_messages) + 1  # 新增1条，而非多条
        print(f"【测试7：重复日志避免】{'通过' if duplicate_check else '失败'} - 新增日志条数正确")

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()