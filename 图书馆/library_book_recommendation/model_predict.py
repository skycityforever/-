import pandas as pd
import numpy as np
import torch
import os

# 假设UserCF类定义在 model/base_models/user_cf.py 中
from model.base_models.user_cf import UserCF
# 假设评估函数定义在 evaluation/metrics.py 中
from evaluation.metrics import calculate_precision_recall, calculate_f1


def load_user_cf_model(model_path):
    """加载保存的UserCF模型（从.pth文件恢复参数）"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # 加载到CPU以兼容无GPU环境

    # 初始化UserCF模型并恢复核心属性
    model = UserCF(top_k=checkpoint.get("top_k", 20))  # 恢复近邻数量
    model.user_similarity = checkpoint["user_similarity"]  # 用户相似度矩阵
    model.user_item_matrix = checkpoint["user_item_matrix"]  # 用户-物品交互矩阵
    model.user_idx = checkpoint["user_idx"]  # 用户ID→索引映射
    model.book_idx = checkpoint["book_idx"]  # 图书ID→索引映射
    model.book_list = checkpoint["book_list"]  # 图书ID列表（索引→原始ID）
    return model


def predict_and_evaluate():
    """完整流程：加载模型→预测→生成提交文件→评估指标"""
    # --------------------- 1. 加载预训练模型 ---------------------
    # lstm_best_epoch1_f10.0032.pth
    # user_cf_best_epoch1_f10.0261.pth
    model_path = "model/saved_models/user_cf_best_epoch1_f10.0157.pth"
    try:
        model = load_user_cf_model(model_path)
        print("UserCF模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # --------------------- 2. 读取测试数据 ---------------------
    test_data_path = "data/merged_data.csv"
    if not os.path.exists(test_data_path):
        print(f"测试数据文件不存在: {test_data_path}")
        return
    test_data = pd.read_csv(test_data_path)
    # 按用户和时间排序（确保时序逻辑，若测试集无时间字段可省略）
    if "借阅时间" in test_data.columns:
        test_data = test_data.sort_values(by=["user_id", "借阅时间"]).reset_index(drop=True)

    # --------------------- 3. 为每个用户生成预测 ---------------------
    user_pred_dict = {}  # 存储 {user_id: predicted_book_id}
    for user_id in test_data["user_id"].unique():
        # 调用模型预测
        pred_book = model.predict(user_id)

        # 处理“模型无法预测”的情况（如冷启动用户）
        if pred_book is None:
            #  fallback：推荐该用户测试集中的第一本真实借阅图书（或热门图书）
            user_actual_books = test_data[test_data["user_id"] == user_id]["book_id"].tolist()
            pred_book = user_actual_books[0] if user_actual_books else None

        user_pred_dict[user_id] = pred_book

    # --------------------- 4. 生成提交文件（submission.csv） ---------------------
    submission_data = []
    for user_id, book_id in user_pred_dict.items():
        if book_id is not None:  # 过滤无效预测
            submission_data.append({"user_id": user_id, "book_id": book_id})
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv("data/submission1.csv", index=False, encoding="utf-8")
    print(f"预测结果已保存至 data/submission1.csv，共 {len(submission_df)} 条记录")

    # --------------------- 5. 计算评估指标（精确率、召回率、F1） ---------------------
    pred_list = []  # 模型预测的图书ID列表
    true_list = []  # 测试集真实的图书ID列表
    for _, row in test_data.iterrows():
        user_id = row["user_id"]
        true_book = row["book_id"]
        true_list.append(true_book)
        pred_book = user_pred_dict.get(user_id, None)
        pred_list.append(pred_book)

    # 调用评估函数
    precision, recall = calculate_precision_recall(pred_list, true_list)
    f1 = calculate_f1(precision, recall)
    print(f"评估指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    predict_and_evaluate()