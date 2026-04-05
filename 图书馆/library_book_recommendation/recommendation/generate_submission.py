import pandas as pd
import torch
import joblib
from ..model.base_models.user_cf import UserCF
from ..model.base_models.wals import WALS
from ..model.advanced_models.lstm_seq_rec import LSTMSeqRecModel, LSTMSeqTrainer
from ..data_preprocessing.data_cleaning import clean_inter_data


def load_trained_model(model_type, model_path, config):
    """加载训练好的模型"""
    if model_type == 'user_cf':
        model = UserCF(top_k=config['user_cf']['top_k'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif model_type == 'wals':
        model = WALS(embedding_dim=config['wals']['embedding_dim'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif model_type == 'lstm':
        user_encoder = joblib.load("model/saved_models/lstm_user_encoder.pkl")
        book_encoder = joblib.load("model/saved_models/lstm_book_encoder.pkl")
        book_num = len(book_encoder.classes_)
        model = LSTMSeqRecModel(
            book_num=book_num,
            embedding_dim=config['lstm']['embedding_dim'],
            hidden_dim=config['lstm']['hidden_dim'],
            num_layers=config['lstm']['num_layers'],
            dropout=config['lstm']['dropout']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, user_encoder, book_encoder
    else:
        raise ValueError("模型类型仅支持'user_cf'、'wals'、'lstm'")


def generate_recommendation(model, test_user_list, book_list, user_idx=None, book_idx=None, model_type='user_cf',
                            lstm_trainer=None, test_inter=None):
    """为测试集用户生成推荐结果（每个用户1本）"""
    recommendations = []
    for user_id in test_user_list:
        if model_type == 'lstm':
            user_inter = test_inter[test_inter['user_id'] == user_id]
            pred_book = lstm_trainer.predict(model, user_inter)
        else:
            pred_book = model.predict(user_id, user_idx, book_idx, book_list)

        if pred_book is not None:
            recommendations.append({'user_id': user_id, 'book_id': pred_book})

    rec_df = pd.DataFrame(recommendations, columns=['user_id', 'book_id'])
    return rec_df


def save_submission(rec_df, save_path):
    """保存提交文件（UTF-8编码，无多余表头）"""
    rec_df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"提交文件已保存至：{save_path}")
    print(f"推荐结果共{len(rec_df)}条，符合赛题测试集用户数量要求")


def main(config):
    # 1. 加载测试集用户列表
    test_inter = pd.read_csv(config['data']['processed_test_path'])
    test_inter = clean_inter_data(config['data']['raw_inter_path'])
    test_user_list = test_inter['user_id'].unique().tolist()

    # 2. 加载图书列表
    book_df = pd.read_csv(config['data']['raw_book_path'])
    book_list = book_df['book_id'].unique().tolist()

    # 3. 加载训练好的模型
    model_type = config['model']['type']
    model_path = config['model']['saved_path']

    if model_type == 'lstm':
        model, user_encoder, book_encoder = load_trained_model(
            model_type=model_type,
            model_path=model_path,
            config=config
        )
        lstm_trainer = LSTMSeqTrainer(config=config)
        lstm_trainer.user_encoder = user_encoder
        lstm_trainer.book_encoder = book_encoder
        # 生成推荐结果
        rec_df = generate_recommendation(
            model=model,
            test_user_list=test_user_list,
            book_list=book_list,
            model_type=model_type,
            lstm_trainer=lstm_trainer,
            test_inter=test_inter
        )
    else:
        model = load_trained_model(
            model_type=model_type,
            model_path=model_path,
            config=config
        )
        # 构建用户和图书的索引（用于UserCF和WALS）
        train_inter = pd.read_csv(config['data']['processed_train_path'])
        user_idx = {user: i for i, user in enumerate(train_inter['user_id'].unique())}
        book_idx = {book: j for j, book in enumerate(book_list)}
        # 生成推荐结果
        rec_df = generate_recommendation(
            model=model,
            test_user_list=test_user_list,
            book_list=book_list,
            user_idx=user_idx,
            book_idx=book_idx,
            model_type=model_type
        )

    # 4. 保存提交文件
    save_submission(rec_df, config['submission']['save_path'])


if __name__ == "__main__":
    from config.config_loader import load_config

    config = load_config('config/config.yaml')
    main(config)