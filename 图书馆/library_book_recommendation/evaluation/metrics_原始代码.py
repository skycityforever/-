import pandas as pd
import numpy as np


def calculate_precision_recall(model, valid_inter, user_list, book_list):
    """计算精确率（P）和召回率（R）{insert\_element\_14\_}"""
    # 构建用户-真实借阅图书字典
    user_true_books = valid_inter.groupby('user_id')['book_id'].apply(set).to_dict()

    tp = 0  # 推荐且用户实际借阅（True Positive）
    fp = 0  # 推荐但用户未借阅（False Positive）
    fn = 0  # 用户借阅但未推荐（False Negative）

    # 遍历验证集中的用户
    for user_id in user_true_books.keys():
        if user_id not in user_list:
            continue  # 跳过训练集中未出现的用户

        # 模型预测推荐图书
        pred_book = model.predict(user_id, model.user_idx, model.book_idx, book_list)
        true_books = user_true_books[user_id]

        if pred_book is None:
            fn += len(true_books)  # 无法预测，视为全部漏推荐
            continue

        # 统计TP、FP、FN
        if pred_book in true_books:
            tp += 1
        else:
            fp += 1
        # 真实借阅但未推荐的数量（用户可能借阅多本，仅推荐1本，故FN=len(true_books)-1 if TP=1 else len(true_books)）
        fn += len(true_books) - (1 if pred_book in true_books else 0)

    # 计算精确率和召回率（避免除以0）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def calculate_f1(precision, recall):
    """计算F1 Score（精确率和召回率的调和平均数）{insert\_element\_15\_}"""
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1
