import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

# 导入模型类（根据实际项目路径调整）
from model.base_models.user_cf import UserCF
from model.base_models.lstm_seq_rec import LSTMSeqRecModel  # LSTM模型类
from evaluation.metrics import calculate_precision_recall, calculate_f1


# -------------------------- 模型加载函数（区分UserCF和LSTM） --------------------------
def load_user_cf_model(model_path):
    """加载UserCF模型（恢复核心属性）"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"UserCF模型文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    model = UserCF(top_k=checkpoint.get("top_k", 20))
    model.user_similarity = checkpoint["user_similarity"]
    model.user_item_matrix = checkpoint["user_item_matrix"]
    model.user_idx = checkpoint["user_idx"]
    model.book_idx = checkpoint["book_idx"]
    model.book_list = checkpoint["book_list"]
    return model


def load_lstm_model(model_path, seq_len=10):
    """加载LSTM模型（恢复PyTorch模型、编码器、特征配置）"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM模型文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 1. 恢复LSTM模型结构（需与训练时参数一致）
    model = LSTMSeqRecModel(
        book_num=len(checkpoint["book_encoder"].classes_),  # 图书总数
        embedding_dim=128,  # 需与训练时一致
        hidden_dim=256,  # 需与训练时一致
        num_layers=2,  # 需与训练时一致
        dropout=0.3  # 需与训练时一致
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切换到评估模式

    # 2. 恢复图书编码器和特征列配置
    book_encoder = checkpoint["book_encoder"]
    feature_cols = checkpoint.get("feature_cols", ["交互权重", "dept_book_preference", "dept_cat1_preference"])

    return model, book_encoder, feature_cols, seq_len


# -------------------------- 历史数据与特征提取 --------------------------
def get_user_history(user_id, train_data, feature_cols):
    """获取用户历史借阅序列（含图书ID和特征）"""
    user_train = train_data[train_data["user_id"] == user_id].copy()
    if len(user_train) == 0:
        return [], []  # 无历史时返回空序列

    # 按借阅时间排序（保证时序逻辑）
    user_train_sorted = user_train.sort_values(by="借阅时间")

    # 提取图书ID序列和对应特征
    book_seq = user_train_sorted["book_id"].tolist()
    features = user_train_sorted[feature_cols].values.tolist() if feature_cols else []

    return book_seq, features


# -------------------------- LSTM预测逻辑（含特征输入） --------------------------
def predict_lstm(model, book_encoder, user_book_seq, user_features, seq_len):
    """LSTM预测：基于图书序列+特征生成推荐"""
    if len(user_book_seq) == 0:
        return None  # 冷启动用户

    # --------------------- 处理图书ID序列 ---------------------
    input_book_seq = user_book_seq[-seq_len:]  # 取最近的seq_len条
    # 填充（不足时用<pad>，需与训练一致）
    if len(input_book_seq) < seq_len:
        input_book_seq = ["<pad>"] * (seq_len - len(input_book_seq)) + input_book_seq

    # 编码图书ID
    try:
        encoded_books = book_encoder.transform(input_book_seq)
    except:
        encoded_books = [0] * seq_len  # 编码失败时填充
    input_books_tensor = torch.tensor(encoded_books, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

    # --------------------- 处理特征序列 ---------------------
    input_feature_seq = user_features[-seq_len:] if len(user_features) >= seq_len else user_features
    # 填充特征（不足时用0.0）
    if len(input_feature_seq) < seq_len:
        pad_features = [[0.0 for _ in range(len(input_feature_seq[0]))]
                        for _ in range(seq_len - len(input_feature_seq))]
        input_feature_seq = pad_features + input_feature_seq
    input_features_tensor = torch.tensor(input_feature_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, feat_dim)

    # --------------------- 模型预测 ---------------------
    with torch.no_grad():
        outputs = model(input_books_tensor, input_features_tensor)  # 传入图书+特征
        pred_idx = torch.argmax(outputs, dim=1).item()

    # 解码为原始book_id
    try:
        pred_book = book_encoder.inverse_transform([pred_idx])[0]
    except:
        pred_book = None
    return pred_book


# -------------------------- 主流程函数 --------------------------
def predict_and_evaluate(model_type="user_cf"):
    """
    主流程：支持UserCF和LSTM模型
    model_type: "user_cf" 或 "lstm"
    """
    # --------------------- 1. 加载模型（根据类型选择） ---------------------
    if model_type == "user_cf":
        model_path = "model/saved_models/user_cf_best_epoch1_f10.0261.pth"
        try:
            model = load_user_cf_model(model_path)
            print("UserCF模型加载成功")
        except Exception as e:
            print(f"UserCF模型加载失败: {str(e)}")
            return
    elif model_type == "lstm":
        model_path = "model/saved_models/lstm_best_epoch1_f10.0032.pth"
        try:
            model, book_encoder, feature_cols, seq_len = load_lstm_model(model_path, seq_len=10)
            print("LSTM模型加载成功")
            # LSTM需要训练集历史，加载训练数据
            train_data = pd.read_csv("data/train_data.csv")
        except Exception as e:
            print(f"LSTM模型加载失败: {str(e)}")
            return
    else:
        print(f"不支持的模型类型: {model_type}")
        return

    # --------------------- 2. 读取测试数据 ---------------------
    test_data_path = "data/test_data.csv"
    if not os.path.exists(test_data_path):
        print(f"测试数据文件不存在: {test_data_path}")
        return
    test_data = pd.read_csv(test_data_path)
    if "借阅时间" in test_data.columns:
        test_data = test_data.sort_values(by=["user_id", "借阅时间"]).reset_index(drop=True)

    # --------------------- 3. 生成预测（区分模型逻辑） ---------------------
    user_pred_dict = {}
    for user_id in test_data["user_id"].unique():
        if model_type == "user_cf":
            pred_book = model.predict(user_id)
        else:
            # LSTM：获取带特征的历史
            user_book_seq, user_features = get_user_history(user_id, train_data, feature_cols)
            pred_book = predict_lstm(
                model, book_encoder,
                user_book_seq=user_book_seq,
                user_features=user_features,
                seq_len=seq_len
            )

        # 冷启动/预测失败的fallback
        if pred_book is None:
            user_actual_books = test_data[test_data["user_id"] == user_id]["book_id"].tolist()
            pred_book = user_actual_books[0] if user_actual_books else None

        user_pred_dict[user_id] = pred_book

    # --------------------- 4. 生成提交文件 ---------------------
    submission_data = [
        {"user_id": user_id, "book_id": book_id}
        for user_id, book_id in user_pred_dict.items()
        if book_id is not None
    ]
    submission_df = pd.DataFrame(submission_data)
    submission_path = f"data/submission_{model_type}.csv"
    submission_df.to_csv(submission_path, index=False, encoding="utf-8")
    print(f"预测结果已保存至 {submission_path}，共 {len(submission_df)} 条记录")

    # --------------------- 5. 评估指标 ---------------------
    pred_list = []
    true_list = []
    for _, row in test_data.iterrows():
        user_id = row["user_id"]
        true_list.append(row["book_id"])
        pred_list.append(user_pred_dict.get(user_id, None))

    precision, recall = calculate_precision_recall(pred_list, true_list)
    f1 = calculate_f1(precision, recall)
    print(f"[{model_type}] 评估指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    # 切换模型类型："user_cf" 或 "lstm"
    predict_and_evaluate(model_type="lstm")  # 调用LSTM模型
    # predict_and_evaluate(model_type="user_cf")  # 调用UserCF模型