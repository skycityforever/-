import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def shuffle_before_split(data, random_state=42):
    """
    数据分割前的洗牌函数：打乱整体数据顺序，降低模型对原始位置的记忆
    参数：
        data: 待洗牌的DataFrame
        random_state: 随机种子（保证结果可复现）
    返回：
        洗牌后的DataFrame
    """
    # 打乱行顺序，保留索引关联
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return shuffled_data


def shuffle_for_model_input(data):
    """
    模型输入时的洗牌函数：每次调用随机打乱，增强训练随机性（无固定种子）
    参数：
        data: 待洗牌的DataFrame（通常是训练集）
    返回：
        随机洗牌后的DataFrame
    """
    # 无固定种子，每次调用生成不同顺序
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    return shuffled_data


def merge_and_split_data(inter_path, item_path, user_path,
                         merged_path, train_path, valid_path, test_path,
                         encoding='utf-8', split_ratios=(0.8, 0.1, 0.1),
                         random_state=42):
    """
    合并三个数据集并分割为训练/验证/测试集（路径参数化）

    参数：
        inter_path: 交互数据表路径（含inter_id, user_id, book_id等）
        item_path: 图书信息表路径（含book_id, 题名, 作者等）
        user_path: 用户信息表路径（含借阅人=user_id, 性别, DEPT等）
        merged_path: 合并后完整数据集的保存路径
        train_path: 训练集保存路径
        valid_path: 验证集保存路径
        test_path: 测试集保存路径
        encoding: 文件编码（如'utf-8', 'gbk'）
        split_ratios: 训练/验证/测试集比例，总和需为1.0
        random_state: 随机种子（保证分割结果可复现）
    """
    # ---------------------- 1. 读取原始数据 ----------------------
    try:
        # 读取交互数据（含inter_id, user_id, book_id等）
        inter_df = pd.read_csv(inter_path, encoding=encoding)
        # 读取图书数据（含book_id, 题名, 作者等）
        item_df = pd.read_csv(item_path, encoding=encoding)
        # 读取用户数据（含“借阅人”=user_id）
        user_df = pd.read_csv(user_path, encoding=encoding)
        print(f"成功读取数据：交互表{len(inter_df)}行 | 图书表{len(item_df)}行 | 用户表{len(user_df)}行")
    except Exception as e:
        raise ValueError(f"读取数据失败：{str(e)}")

    # ---------------------- 2. 数据合并（外键关联） ----------------------
    # 重命名用户表的“借阅人”为“user_id”，实现外键关联
    user_df_renamed = user_df.rename(columns={"借阅人": "user_id"})

    # 第一步：交互表 + 图书表（通过book_id关联）
    merged_step1 = pd.merge(
        inter_df,
        item_df,
        on="book_id",
        how="left"  # 保留所有交互记录，图书信息缺失则为NaN
    )

    # 第二步：合并用户信息（通过user_id关联）
    merged_full = pd.merge(
        merged_step1,
        user_df_renamed,
        on="user_id",
        how="left"  # 保留所有交互记录，用户信息缺失则为NaN
    )

    # ---------------------- 3. 调整列顺序 ----------------------
    target_columns = [
        "inter_id", "user_id", "book_id", "借阅时间", "还书时间", "续借时间", "续借次数",
        "题名", "作者", "出版社", "一级分类", "二级分类",
        "性别", "DEPT", "年级", "类型"
    ]
    # 检查必要列是否存在（避免因列名不符导致报错）
    missing_cols = [col for col in target_columns if col not in merged_full.columns]
    if missing_cols:
        raise ValueError(f"合并后的数据缺失必要列：{missing_cols}")
    merged_full = merged_full[target_columns]

    # ---------------------- 4. 分割前洗牌（降低位置记忆） ----------------------
    merged_shuffled = shuffle_before_split(merged_full, random_state=random_state)
    print(f"数据合并完成，共{len(merged_shuffled)}行，已完成分割前洗牌")

    # ---------------------- 5. 按比例分割数据 ----------------------
    train_ratio, valid_ratio, test_ratio = split_ratios
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("分割比例总和必须为1.0")

    # 第一步：分割训练集和剩余数据
    train_size = int(len(merged_shuffled) * train_ratio)
    train = merged_shuffled.iloc[:train_size]
    remaining = merged_shuffled.iloc[train_size:]

    # 第二步：分割验证集和测试集
    valid_size = int(len(merged_shuffled) * valid_ratio)
    valid = remaining.iloc[:valid_size]
    test = remaining.iloc[valid_size:]

    # ---------------------- 6. 保存数据 ----------------------
    merged_shuffled.to_csv(merged_path, index=False, encoding=encoding)
    train.to_csv(train_path, index=False, encoding=encoding)
    valid.to_csv(valid_path, index=False, encoding=encoding)
    test.to_csv(test_path, index=False, encoding=encoding)

    print(f"数据分割完成：")
    print(f"训练集：{len(train)}行（{train_ratio * 100}%） | 保存至 {train_path}")
    print(f"验证集：{len(valid)}行（{valid_ratio * 100}%） | 保存至 {valid_path}")
    print(f"测试集：{len(test)}行（{test_ratio * 100}%） | 保存至 {test_path}")
    print(f"合并后完整数据：{len(merged_shuffled)}行 | 保存至 {merged_path}")


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 虚拟路径（实际使用时替换为真实路径）
    INPUT_PATHS = {
        "inter": "../data/inter_preliminary.csv",  # 交互数据路径
        "item": "../data/item.csv",  # 图书数据路径
        "user": "../data/user.csv"  # 用户数据路径
    }
    OUTPUT_PATHS = {
        "merged": "../data/merged_data.csv",  # 合并后完整数据
        "train": "../data/train_data.csv",  # 训练集
        "valid": "../data/valid_data.csv",  # 验证集
        "test": "../data/test_data.csv"  # 测试集
    }


    # 调用合并分割函数（若文件编码为GBK，可将encoding改为'gbk'）
    merge_and_split_data(
        inter_path=INPUT_PATHS["inter"],
        item_path=INPUT_PATHS["item"],
        user_path=INPUT_PATHS["user"],
        merged_path=OUTPUT_PATHS["merged"],
        train_path=OUTPUT_PATHS["train"],
        valid_path=OUTPUT_PATHS["valid"],
        test_path=OUTPUT_PATHS["test"],
        encoding='utf-8',  # 若中文乱码，尝试改为'gbk'
        split_ratios=(0.6, 0.2, 0.2),  # 6:2:2分割
        random_state=42
    )

    # 测试模型输入时的洗牌函数（以训练集为例）
    train_data = pd.read_csv(OUTPUT_PATHS["train"], encoding='utf-8')
    train_shuffled = shuffle_for_model_input(train_data)
    print(f"\n模型输入洗牌示例：原始训练集前3行索引 vs 洗牌后前3行索引")
    print(f"原始：{train_data.index[:3].tolist()} | 洗牌后：{train_shuffled.index[:3].tolist()}")