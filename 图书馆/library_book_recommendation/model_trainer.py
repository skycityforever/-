import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import random
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 数据预处理模块导入
from 图书馆.library_book_recommendation.data_preprocessing.data_cleaning import clean_book_secondary_category, \
    clean_inter_data
from 图书馆.library_book_recommendation.data_preprocessing.data_splitting import merge_and_split_data
from 图书馆.library_book_recommendation.data_preprocessing.feature_engineering import (
    extract_time_features,
    build_interaction_weight,
    encode_categorical_features,
    merge_dept_features_to_inter,
    add_user_dept_features
)
from transformers import BertTokenizer

# 模型和工具导入
from 图书馆.library_book_recommendation.model.word2vec import BookTitleWord2Vec
from 图书馆.library_book_recommendation.model.advanced_models.bert import BertSeqRecModel, BertBorrowDataset
from 图书馆.library_book_recommendation.model.base_models.user_cf import UserCF
from 图书馆.library_book_recommendation.model.advanced_models.lstm_seq_rec import LSTMSeqRecModel, BorrowSequenceDataset
from 图书馆.library_book_recommendation.utils.logger import Logger
from 图书馆.library_book_recommendation.evaluation.metrics import calculate_precision_recall, calculate_f1


class EnhancedModelTrainer:
    def __init__(self, data_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化训练器（新增书名向量相关属性）"""
        # 原有属性
        self.test_data = None
        self.book_cat_encoder = None
        self.user_encoder = None
        self.device = self._init_device(device)
        self.logger = Logger('enhanced_training.log').get_logger()
        self.logger.info(f"训练设备初始化完成: {self.device}")
        self.data_paths = data_paths
        self.cleaned_data = None
        self.train_data = None
        self.valid_data = None
        self.loss_history = {}
        self.metric_history = {}
        self.best_models = {}
        self.best_metrics = {}
        self.item_data = data_paths['books']
        # LSTM预测相关属性
        self.lstm_model = None  # 加载的LSTM模型
        self.lstm_user_encoder = None  # LSTM用户编码器
        self.lstm_book_encoder = None  # LSTM图书编码器
        self.lstm_seq_len = 7  # LSTM序列长度
        self.lstm_title_vec_dim = 0  # 书名向量维度
        self.title_vector_path = "data/item_with_title_vector.csv"  # 书名向量保存路径

    def _init_device(self, device):
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，自动切换到CPU")
            return 'cpu'
        return device

    def load_and_preprocess_data(self, test_size=0.1, random_state=42, train_word2vec=True):
        """修改：新增书名向量生成步骤"""
        self.logger.info("开始数据预处理...")
        start_time = time.time()

        # 清洗图书数据并保存
        books_clean = clean_book_secondary_category(self.data_paths['books'])
        cleaned_books_path = "data/cleaned_books.csv"
        books_clean.to_csv(cleaned_books_path, index=False, encoding='utf-8')
        self.logger.info(f"图书数据清洗完成并保存: {len(books_clean)}条 -> {cleaned_books_path}")
        self.item_data = books_clean
        # 训练Word2Vec生成书名向量
        if train_word2vec:
            self.logger.info("开始训练Word2Vec生成书名向量...")
            w2v = BookTitleWord2Vec(
                vector_size=128,  # 可调整向量维度
                window=3,
                min_count=1,
                workers=4
            )
            # 从清洗后的图书数据中读取书名
            titles = books_clean['书名'].dropna().unique().tolist()
            # 训练并生成含向量的图书数据
            w2v.train(titles, model_save_path="model/saved_models/title_word2vec.model")
            # 为item.csv生成书名向量并保存
            w2v.generate_title_vectors_for_csv(
                input_csv_path=cleaned_books_path,
                output_csv_path=self.title_vector_path,
                title_col="书名",
                id_col="book_id"
            )
            self.logger.info(f"书名向量生成完成并保存至: {self.title_vector_path}")

        # 清洗借阅数据并保存
        borrows_clean = clean_inter_data(self.data_paths['borrows'])
        cleaned_borrows_path = "data/cleaned_borrows.csv"
        borrows_clean.to_csv(cleaned_borrows_path, index=False, encoding='utf-8')
        self.logger.info(f"借阅数据清洗完成并保存: {len(borrows_clean)}条 -> {cleaned_borrows_path}")

        # 合并分割数据（原有逻辑不变）
        merged_path = "data/merged_data.csv"
        train_path = "data/train_data.csv"
        valid_path = "data/valid_data.csv"
        test_path = "data/test_data.csv"
        merge_and_split_data(
            inter_path=cleaned_borrows_path,
            item_path=cleaned_books_path,
            user_path=self.data_paths['users'],
            merged_path=merged_path,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            encoding='utf-8',
            split_ratios=(0.8, 0.1, 0.1),
            random_state=random_state
        )

        # 训练集、验证集、测试集特征工程（原有逻辑不变）
        # 训练集特征工程
        train_data = pd.read_csv(train_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        train_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        train_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')
        train_inter_enhanced = extract_time_features(train_data.copy())
        train_inter_enhanced = build_interaction_weight(train_inter_enhanced)
        train_users_encoded, train_books_encoded = encode_categorical_features(train_users.copy(), train_books.copy())
        self.user_encoder = train_users_encoded['DEPT'].astype('category').cat.categories
        self.book_cat_encoder = train_books_encoded['二级分类'].astype('category').cat.categories
        train_inter_enhanced = merge_dept_features_to_inter(train_inter_enhanced.copy(), train_users.copy(),
                                                            train_books.copy())
        train_users_enhanced = add_user_dept_features(train_users_encoded.copy(), train_inter_enhanced.copy(),
                                                      train_books.copy())
        train_inter_enhanced.to_csv("data/cleaned_borrows_enhanced.csv", index=False, encoding='utf-8')
        train_books_encoded.to_csv("data/cleaned_books_encoded.csv", index=False, encoding='utf-8')
        train_users_enhanced.to_csv("data/cleaned_user_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"训练集特征工程完成: {len(train_inter_enhanced)}条 -> cleaned_borrows_enhanced.csv")

        # 验证集特征工程
        valid_data = pd.read_csv(valid_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        valid_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        valid_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')
        valid_inter_enhanced = extract_time_features(valid_data.copy())
        valid_inter_enhanced = build_interaction_weight(valid_inter_enhanced)
        valid_users_encoded = valid_users.copy().rename(columns={"借阅人": "user_id"})
        valid_users_encoded['DEPT_encoded'] = valid_users_encoded['DEPT'].astype('category').cat.set_categories(
            self.user_encoder).cat.codes
        valid_books_encoded = pd.get_dummies(valid_books, columns=['一级分类'], prefix='book_cat1')
        valid_books_encoded['二级分类_encoded'] = valid_books_encoded['二级分类'].astype('category').cat.set_categories(
            self.book_cat_encoder).cat.codes
        valid_inter_enhanced = merge_dept_features_to_inter(valid_inter_enhanced.copy(), valid_users.copy(),
                                                            valid_books.copy())
        valid_users_enhanced = add_user_dept_features(valid_users_encoded.copy(), valid_inter_enhanced.copy(),
                                                      valid_books.copy())
        valid_inter_enhanced.to_csv("data/valid_borrows_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"验证集特征处理完成: {len(valid_inter_enhanced)}条 -> valid_borrows_enhanced.csv")

        # 测试集特征工程
        test_data = pd.read_csv(test_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        test_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        test_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')
        test_inter_enhanced = extract_time_features(test_data.copy())
        test_inter_enhanced = build_interaction_weight(test_inter_enhanced)
        test_users_encoded = test_users.copy().rename(columns={"借阅人": "user_id"})
        test_users_encoded['DEPT_encoded'] = test_users_encoded['DEPT'].astype('category').cat.set_categories(
            self.user_encoder).cat.codes
        test_books_encoded = pd.get_dummies(test_books, columns=['一级分类'], prefix='book_cat1')
        test_books_encoded['二级分类_encoded'] = test_books_encoded['二级分类'].astype('category').cat.set_categories(
            self.book_cat_encoder).cat.codes
        test_inter_enhanced = merge_dept_features_to_inter(test_inter_enhanced.copy(), test_users.copy(),
                                                           test_books.copy())
        test_users_enhanced = add_user_dept_features(test_users_encoded.copy(), test_inter_enhanced.copy(),
                                                     test_books.copy())
        test_inter_enhanced.to_csv("data/test_borrows_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"测试集特征处理完成: {len(test_inter_enhanced)}条 -> test_borrows_enhanced.csv")

        # 总数据集特征工程
        merged_data = pd.read_csv(merged_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        merged_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        merged_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')
        merged_inter_enhanced = extract_time_features(merged_data.copy())
        merged_inter_enhanced = build_interaction_weight(merged_inter_enhanced)
        merged_users_encoded = test_users.copy().rename(columns={"借阅人": "user_id"})
        merged_users_encoded['DEPT_encoded'] = merged_users_encoded['DEPT'].astype('category').cat.set_categories(
            self.user_encoder).cat.codes
        merged_books_encoded = pd.get_dummies(merged_books, columns=['一级分类'], prefix='book_cat1')
        merged_books_encoded['二级分类_encoded'] = merged_books_encoded['二级分类'].astype('category').cat.set_categories(
            self.book_cat_encoder).cat.codes
        merged_inter_enhanced = merge_dept_features_to_inter(merged_inter_enhanced.copy(), merged_users.copy(),
                                                           merged_books.copy())
        merged_users_enhanced = add_user_dept_features(merged_users_encoded.copy(), merged_inter_enhanced.copy(),
                                                     merged_books.copy())
        merged_inter_enhanced.to_csv("data/merged_borrows_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"总数据集特征处理完成: {len(merged_inter_enhanced)}条 -> merged_borrows_enhanced.csv")


        # 加载分割后的数据
        self.train_data = pd.read_csv("data/merged_borrows_enhanced.csv", encoding='utf-8')
        self.valid_data = pd.read_csv("data/valid_borrows_enhanced.csv", encoding='utf-8')
        self.test_data = pd.read_csv("data/merged_borrows_enhanced.csv", encoding='utf-8')
        self.logger.info(
            f"数据分割完成 - 训练集: {len(self.train_data)}条, 验证集: {len(self.valid_data)}条, 测试集: {len(self.test_data)}条")
        self.logger.info(f"数据预处理总耗时: {time.time() - start_time:.2f}秒")

    def _get_user_input(self, model_type):
        """原有参数输入逻辑（保持不变）"""
        params = {}
        params['k_fold'] = int(input("是否使用K折交叉验证？(1=是,0=否): "))
        if params['k_fold']:
            params['n_splits'] = int(input("请输入折数(默认5): ") or 5)

        if model_type == 'lstm':
            default_epochs = 30
            params['epochs'] = int(input(f"请输入训练轮数(默认{default_epochs}): ") or default_epochs)

        if model_type == 'user_cf':
            params['top_k'] = int(input("请输入近邻数量(默认20): ") or 20)
        elif model_type == 'lstm':
            params['seq_len'] = int(input("请输入序列长度(默认5): ") or 5)
            params['embedding_dim'] = int(input("请输入嵌入维度(默认128): ") or 128)
            params['hidden_dim'] = int(input("请输入隐藏层维度(默认256): ") or 256)
            params['num_layers'] = int(input("请输入LSTM层数(默认2): ") or 2)
            params['dropout'] = float(input("请输入dropout比例(默认0.3): ") or 0.3)
            params['batch_size'] = int(input("请输入批次大小(默认32): ") or 32)
            params['lr'] = float(input("请输入学习率(默认0.001): ") or 0.001)
            params['patience'] = int(input("请输入早停耐心值(默认3): ") or 3)
        elif model_type == 'hybrid':
            params['cf_weight'] = float(input("请输入协同过滤权重(0-1,默认0.5): ") or 0.5)
            params['lstm_weight'] = 1 - params['cf_weight']
            print("=== 请配置协同过滤子模型参数 ===")
            params['cf_params'] = self._get_user_input('user_cf')
            print("=== 请配置LSTM子模型参数 ===")
            params['lstm_params'] = self._get_user_input('lstm')

        return params

    def _visualize_step(self, model_type, epoch, loss, acc, f1, epoch_time):
        """原有可视化逻辑（保持不变）"""
        print(f"\rEpoch {epoch:3d} - "
              f"损失: {loss:.4f} | "
              f"准确率: {acc:.4f} | "
              f"F1分数: {f1:.4f} | "
              f"用时: {epoch_time:.2f}秒", end='')

    def _visualize_summary(self, model_type):
        """原有可视化总结（保持不变）"""
        if model_type not in self.loss_history:
            print("无训练历史可可视化")
            return

        print("\n" + "=" * 60)
        print(f"{model_type}模型训练完整记录")
        print("=" * 60)

        loss_data = self.loss_history[model_type]
        max_loss, min_loss = max(loss_data), min(loss_data)
        print("\n损失变化趋势:")
        for i, loss in enumerate(loss_data):
            bar_len = int(30 * (1 - (loss - min_loss) / (max_loss - min_loss + 1e-8)))
            print(f"Epoch {i + 1:3d}: {loss:.4f} " + "|" + "*" * bar_len + " " * (30 - bar_len) + "|")

        if model_type in self.metric_history:
            f1_data = [m['f1'] for m in self.metric_history[model_type]]
            print("\nF1分数变化趋势:")
            for i, f1 in enumerate(f1_data):
                bar_len = int(30 * f1)
                print(f"Epoch {i + 1:3d}: {f1:.4f} " + "|" + "*" * bar_len + " " * (30 - bar_len) + "|")

    def train_bert(self, params):
        """训练BERT+词向量融合模型"""
        model_type = 'bert'
        start_time = time.time()
        print("\n=== 开始训练BERT+词向量模型 ===")
        self.logger.info("开始训练BERT+词向量模型")

        # 1. 初始化编码器和分词器
        user_encoder = LabelEncoder()
        book_encoder = LabelEncoder()
        user_encoder.fit(self.train_data['user_id'].unique())
        all_book_ids = pd.concat([
            self.train_data['book_id'],
            self.valid_data['book_id'],
            self.test_data['book_id']
        ]).unique()
        book_encoder.fit(all_book_ids)
        book_num = len(book_encoder.classes_)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # 2. 创建数据集和数据加载器
        train_dataset = BertBorrowDataset(
            self.train_data,
            user_encoder,
            book_encoder,
            seq_len=params['seq_len'],
            title_vector_path=self.title_vector_path,
            bert_tokenizer=bert_tokenizer,
            max_bert_len=params['max_bert_len']
        )
        valid_dataset = BertBorrowDataset(
            self.valid_data,
            user_encoder,
            book_encoder,
            seq_len=params['seq_len'],
            title_vector_path=self.title_vector_path,
            bert_tokenizer=bert_tokenizer,
            max_bert_len=params['max_bert_len']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=2  # 多进程加载
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=2
        )

        # 3. 初始化模型
        model = BertSeqRecModel(
            book_num=book_num,
            feature_dim=3,  # 基础特征维度（与数据集对应）
            title_vec_dim=train_dataset.title_vec_dim,  # 词向量维度（从数据集获取）
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            freeze_bert=params.get('freeze_bert', False)  # 是否冻结BERT
        ).to(self.device)
        self.logger.info(f"BERT模型已加载至设备: {self.device}，词向量维度: {train_dataset.title_vec_dim}")

        # 4. 定义损失函数和优化器（BERT建议用较小学习率）
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=1e-5  # 权重衰减防过拟合
        )

        # 5. 训练循环
        self.loss_history[model_type] = []
        self.metric_history[model_type] = []
        best_f1 = 0.0
        no_improve_epochs = 0

        for epoch in range(1, params['epochs'] + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            train_preds = []
            train_trues = []

            for _, input_ids, attention_mask, title_vecs, features, target in tqdm(
                    train_loader, desc=f"Epoch {epoch}/{params['epochs']}", leave=False
            ):
                # 转移到设备
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                title_vecs = title_vecs.to(self.device)
                features = features.to(self.device)
                target = target.squeeze().to(self.device)  # [batch_size]

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, title_vecs, features)  # [batch_size, book_num]
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_trues.extend(target.cpu().numpy())

            # 计算训练集指标
            avg_loss = total_loss / len(train_loader)
            train_precision, train_recall = self._calculate_sequence_metrics(train_preds, train_trues)
            train_f1 = calculate_f1(train_precision, train_recall)

            # 验证集评估
            model.eval()
            valid_preds = []
            valid_trues = []
            with torch.no_grad():
                for _, input_ids, attention_mask, title_vecs, features, target in valid_loader:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    title_vecs = title_vecs.to(self.device)
                    features = features.to(self.device)
                    target = target.squeeze().to(self.device)

                    outputs = model(input_ids, attention_mask, title_vecs, features)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    valid_preds.extend(preds)
                    valid_trues.extend(target.cpu().numpy())

            valid_precision, valid_recall = self._calculate_sequence_metrics(valid_preds, valid_trues)
            valid_f1 = calculate_f1(valid_precision, valid_recall)

            # 记录和可视化
            self.loss_history[model_type].append(avg_loss)
            self.metric_history[model_type].append({
                'train_f1': train_f1, 'valid_f1': valid_f1, 'epoch': epoch
            })
            epoch_time = time.time() - epoch_start
            self._visualize_step(model_type, epoch, avg_loss, valid_precision, valid_f1, epoch_time)

            # 早停和保存最佳模型
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                self._save_best_model(
                    model, model_type, epoch, best_f1,
                    user_encoder=user_encoder,
                    book_encoder=book_encoder,
                    seq_len=params['seq_len'],
                    hidden_dim=params['hidden_dim'],
                    dropout=params['dropout'],
                    title_vector_dim=train_dataset.title_vec_dim,
                    max_bert_len=params['max_bert_len']
                )
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= params['patience']:
                    self.logger.info(f"早停触发: 连续{params['patience']}轮无提升")
                    break

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒，最佳验证集F1: {best_f1:.4f}")
        self.logger.info(f"BERT模型训练完成 - 最佳F1: {best_f1:.4f}，耗时: {total_time:.2f}秒")
        return model, user_encoder, book_encoder, best_f1

    def train_user_cf(self, params):
        """原有UserCF训练逻辑（保持不变）"""
        model_type = 'user_cf'
        start_time = time.time()
        print("\n=== 开始训练用户协同过滤模型 ===")
        self.logger.info("开始训练用户协同过滤模型")

        model = UserCF(top_k=params['top_k'])
        model.fit(self.train_data)

        precision, recall = calculate_precision_recall(model, self.valid_data)
        f1_score = calculate_f1(precision, recall)
        acc = precision

        self.loss_history[model_type] = [0.0]
        self.metric_history[model_type] = [{'precision': precision, 'recall': recall, 'f1': f1_score, 'acc': acc}]

        self._save_best_model(model, model_type, 1, f1_score)

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        print(f"验证集指标 - 准确率: {acc:.4f}, F1: {f1_score:.4f}")
        self.logger.info(f"用户协同过滤训练完成 - 准确率: {acc:.4f}, F1: {f1_score:.4f}, 耗时: {total_time:.2f}秒")

        return model, f1_score

    def train_lstm(self, params):
        """修改：适配书名向量特征的LSTM训练逻辑"""
        model_type = 'lstm'
        start_time = time.time()
        print("\n=== 开始训练LSTM模型 ===")
        self.logger.info("开始训练LSTM模型")

        user_encoder = LabelEncoder()
        book_encoder = LabelEncoder()
        user_encoder.fit(self.train_data['user_id'].unique())
        all_book_ids = pd.concat([
            self.item_data['book_id']
        ]).unique()
        book_encoder.fit(all_book_ids)
        book_num = len(book_encoder.classes_)

        # 修改1：创建数据集时传入书名向量路径
        train_dataset = BorrowSequenceDataset(
            self.train_data, user_encoder, book_encoder,
            seq_len=params['seq_len'],
            title_vector_path=self.title_vector_path  # 新增：传入书名向量路径
        )
        valid_dataset = BorrowSequenceDataset(
            self.valid_data, user_encoder, book_encoder,
            seq_len=params['seq_len'],
            title_vector_path=self.title_vector_path  # 新增：传入书名向量路径
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])

        # 修改2：初始化模型时传入书名向量维度
        title_vector_dim = train_dataset.title_vector_dim  # 从数据集获取向量维度
        model = LSTMSeqRecModel(
            book_num=book_num,
            feature_dim=3,  # 原有基础特征维度
            title_vector_dim=title_vector_dim,  # 新增：书名向量维度
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        self.logger.info(f"LSTM模型已加载至设备: {self.device}，书名向量维度: {title_vector_dim }")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        self.loss_history[model_type] = []
        self.metric_history[model_type] = []
        best_f1 = 0.0
        no_improve_epochs = 0

        for epoch in range(1, params['epochs'] + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            train_preds = []
            train_trues = []

            for input_ids, input_features, target in tqdm(train_loader, desc=f"Epoch {epoch}/{params['epochs']}",
                                                          leave=False):
                input_ids = input_ids.to(self.device)
                input_features = input_features.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, input_features)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_trues.extend(target.cpu().numpy())

            avg_loss = total_loss / len(train_loader)
            train_precision, train_recall = self._calculate_sequence_metrics(train_preds, train_trues)
            train_f1 = calculate_f1(train_precision, train_recall)
            train_acc = train_precision

            model.eval()
            with torch.no_grad():
                valid_preds = []
                valid_trues = []
                for input_ids, input_features, target in valid_loader:
                    input_ids = input_ids.to(self.device)
                    input_features = input_features.to(self.device)
                    outputs = model(input_ids, input_features)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    valid_preds.extend(preds)
                    valid_trues.extend(target.cpu().numpy())

            valid_precision, valid_recall = self._calculate_sequence_metrics(valid_preds, valid_trues)
            valid_f1 = calculate_f1(valid_precision, valid_recall)
            valid_acc = valid_precision

            self.loss_history[model_type].append(avg_loss)
            self.metric_history[model_type].append({
                'train_acc': train_acc, 'train_f1': train_f1,
                'valid_acc': valid_acc, 'valid_f1': valid_f1,
                'epoch': epoch
            })

            epoch_time = time.time() - epoch_start
            self._visualize_step(model_type, epoch, avg_loss, valid_acc, valid_f1, epoch_time)

            if valid_f1 > best_f1:
                best_f1 = valid_f1
                # 修改3：保存模型时记录书名向量维度
                self._save_best_model(model, model_type, epoch, best_f1,
                                      user_encoder=user_encoder, book_encoder=book_encoder,
                                      seq_len=params['seq_len'],
                                      embedding_dim=params['embedding_dim'],
                                      hidden_dim=params['hidden_dim'],
                                      num_layers=params['num_layers'],
                                      dropout=params['dropout'],
                                      title_vector_dim=title_vector_dim  # 新增：保存书名向量维度
                                      )
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= params['patience']:
                    self.logger.info(f"早停触发: 连续{params['patience']}轮无提升")
                    break

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        print(f"最佳验证集F1: {best_f1:.4f}")
        self.logger.info(f"LSTM训练完成 - 最佳F1: {best_f1:.4f}, 总耗时: {total_time:.2f}秒")

        return model, user_encoder, book_encoder, best_f1

    def train_hybrid(self, params):
        """修改：混合模型适配书名向量"""
        model_type = 'hybrid'
        start_time = time.time()
        print("\n=== 开始训练混合模型 ===")
        self.logger.info("开始训练混合模型")

        cf_model, cf_f1 = self.train_user_cf(params['cf_params'])
        lstm_model, user_encoder, book_encoder, lstm_f1 = self.train_lstm(params['lstm_params'])

        class HybridModel:
            def __init__(self, cf_model, lstm_model, user_encoder, book_encoder,
                         cf_weight=0.5, lstm_weight=0.5, seq_len=5, title_vector_path=None, trainer=None):
                self.cf_model = cf_model
                self.lstm_model = lstm_model
                self.user_encoder = user_encoder
                self.book_encoder = book_encoder
                self.cf_weight = cf_weight
                self.lstm_weight = lstm_weight
                self.seq_len = seq_len
                self.trainer = trainer
                self.title_vector_path = title_vector_path  # 新增：书名向量路径

            def predict(self, user_id, user_idx, book_idx, book_list, user_inter=None):
                cf_rec = self.cf_model.predict(user_id, user_idx, book_idx, book_list)
                if user_inter is not None and len(user_inter) >= self.seq_len:
                    lstm_rec = self.trainer._lstm_predict(
                        self.lstm_model, user_inter, self.user_encoder,
                        self.book_encoder, self.seq_len, self.title_vector_path  # 新增：传入书名向量路径
                    )
                    lstm_rec = lstm_rec if lstm_rec is not None else cf_rec
                else:
                    lstm_rec = cf_rec
                return cf_rec if random.random() < self.cf_weight else lstm_rec

        hybrid_model = HybridModel(
            cf_model, lstm_model, user_encoder, book_encoder,
            params['cf_weight'], params['lstm_weight'],
            params['lstm_params']['seq_len'],
            title_vector_path=self.title_vector_path,  # 新增：传入书名向量路径
            trainer=self
        )

        user_list = self.train_data['user_id'].unique().tolist()
        book_list = self.train_data['book_id'].unique().tolist()
        user_idx = {u: i for i, u in enumerate(user_list)}
        book_idx = {b: i for i, b in enumerate(book_list)}

        precision, recall = calculate_precision_recall(
            hybrid_model, self.valid_data, user_list, book_list
        )
        f1_score = calculate_f1(precision, recall)
        acc = precision

        self._save_best_model(hybrid_model, model_type, 1, f1_score)

        total_time = time.time() - start_time
        print(f"\n混合模型训练完成! 总耗时: {total_time:.2f}秒")
        print(f"验证集指标 - 准确率: {acc:.4f}, F1: {f1_score:.4f}")
        self.logger.info(f"混合模型训练完成 - 准确率: {acc:.4f}, F1: {f1_score:.4f}, 耗时: {total_time:.2f}秒")

        return hybrid_model, f1_score

    def calculate_precision_recall(model, valid_data):
        """原有评估逻辑（保持不变）"""
        hits = 0
        total_pred = 0
        total_true = 0

        for user_id, user_group in valid_data.groupby('user_id'):
            if user_id not in model.user_idx:
                continue

            true_books = set(user_group['book_id'].tolist())
            if not true_books:
                continue
            total_true += len(true_books)

            pred_book = model.predict(user_id)
            total_pred += 1

            if pred_book in true_books:
                hits += 1

        precision = hits / total_pred if total_pred > 0 else 0.0
        recall = hits / total_true if total_true > 0 else 0.0

        return precision, recall

    def _calculate_sequence_metrics(self, preds, trues):
        """原有序列评估逻辑（保持不变）"""
        tp = sum(p == t for p, t in zip(preds, trues))
        fp = len(preds) - tp
        fn = len(trues) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def _lstm_predict(self, model, user_inter, user_encoder, book_encoder, seq_len, title_vector_path=None):
        """修改：返回预测书籍ID + 预测置信度（概率），用于阈值判断"""
        try:
            book_ids = [inter['book_id'] for inter in user_inter]
            # 原有基础特征
            base_features = [
                [inter['交互权重'], inter['dept_book_preference'], inter['dept_cat1_preference']]
                for inter in user_inter
            ]

            # 加载书名向量并拼接
            if title_vector_path and os.path.exists(title_vector_path):
                vec_df = pd.read_csv(title_vector_path)
                vector_cols = [col for col in vec_df.columns if col.startswith('title_vec_')]
                title_vec_dict = {
                    str(row['book_id']): row[vector_cols].values
                    for _, row in vec_df.iterrows()
                }
                # 拼接书名向量
                for i, book_id in enumerate(book_ids):
                    book_id_str = str(book_id)
                    if book_id_str in title_vec_dict:
                        base_features[i] = np.concatenate([base_features[i], title_vec_dict[book_id_str]], axis=0)
                    else:
                        vec_dim = len(vector_cols) if vector_cols else 0
                        base_features[i] = np.concatenate([base_features[i], np.zeros(vec_dim)], axis=0)

            # 编码图书ID并处理序列长度
            encoded_books = book_encoder.transform(book_ids)
            input_ids = encoded_books[-seq_len:].astype(np.int64)
            input_features = np.array(base_features[-seq_len:], dtype=np.float32)

            if len(input_ids) < seq_len:
                pad_length = seq_len - len(input_ids)
                input_ids = np.pad(input_ids, (pad_length, 0), mode='constant')
                input_features = np.pad(input_features, ((pad_length, 0), (0, 0)), mode='constant')

            # 转换为张量并预测
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            input_features_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            model.eval()
            with torch.no_grad():
                output = model(input_ids_tensor, input_features_tensor)  # 模型输出logits
                # 新增：计算预测置信度（logits→softmax→概率）
                output_softmax = torch.softmax(output, dim=1)  # 转为概率分布
                pred_idx = torch.argmax(output_softmax, dim=1).cpu().item()  # 概率最大的类别索引
                pred_confidence = output_softmax[0, pred_idx].cpu().item()  # 对应类别的置信度（0-1之间）

            # 新增：返回（预测书籍ID，预测置信度）
            pred_book = book_encoder.inverse_transform([pred_idx])[0]
            return pred_book, pred_confidence
        except Exception as e:
            self.logger.error(f"LSTM预测错误: {str(e)}")
            return None, 0.0  # 预测失败时，置信度返回0

    def _save_best_model(self, model, model_type, epoch, f1_score, **kwargs):
        """修改：保存模型时包含书名向量维度"""
        save_dir = "model/saved_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{model_type}_best_epoch{epoch}_f1{f1_score:.4f}.pth"

        save_data = {"f1_score": f1_score, "epoch": epoch}
        save_data.update(kwargs)  # 包含title_vec_dim（如果存在）

        if isinstance(model, torch.nn.Module):
            save_data["model_state_dict"] = model.state_dict()
        else:
            save_data.update({
                "user_similarity": model.user_similarity,
                "user_item_matrix": model.user_item_matrix,
                "user_idx": model.user_idx,
                "book_idx": model.book_idx,
                "book_list": model.book_list,
                "top_k": model.top_k
            })

        torch.save(save_data, save_path)
        self.best_models[model_type] = save_path
        self.best_metrics[model_type] = f1_score
        self.logger.info(f"最佳{model_type}模型已保存至: {save_path} (F1: {f1_score:.4f})")

    # -------------------------- 新增：LSTM预测相关方法修改 --------------------------
    def load_lstm_model(self, model_path):
        """修改：加载模型时恢复书名向量维度"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))

        # 恢复编码器
        self.lstm_user_encoder = checkpoint.get("user_encoder")
        self.lstm_book_encoder = checkpoint.get("book_encoder")
        if self.lstm_user_encoder is None or self.lstm_book_encoder is None:
            raise ValueError("模型文件中未找到用户/图书编码器")

        # 恢复模型超参数（新增书名向量维度）
        self.lstm_seq_len = checkpoint.get("seq_len", 7)
        embedding_dim = 128  # 强制与旧模型一致
        hidden_dim = 256  # 从错误信息1024=4*256反推
        # self.lstm_title_vec_dim = checkpoint.get("title_vec_dim", 0)  # 恢复书名向量维度
        # embedding_dim = checkpoint.get("embedding_dim", 128)
        # hidden_dim = checkpoint.get("hidden_dim", 256)
        num_layers = checkpoint.get("num_layers", 1)
        dropout = checkpoint.get("dropout", 0.1)
        book_num = len(self.lstm_book_encoder.classes_)
        title_vector_dim = 259 - embedding_dim - 3  # 128
        # 初始化并加载模型权重（传入书名向量维度）
        self.lstm_model = LSTMSeqRecModel(
            book_num=book_num,
            feature_dim=3,
            title_vector_dim= title_vector_dim,  # 新增：传入书名向量维度
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self.lstm_model.eval()  # 设为评估模式

        self.logger.info(f"LSTM模型加载成功: {model_path}，书名向量维度: {title_vector_dim}")

    def _get_bert_params(self):
        params = {}
        params['k_fold'] = int(input("是否使用K折交叉验证？(1=是,0=否): "))
        if params['k_fold']:
            params['n_splits'] = int(input("请输入折数(默认5): ") or 5)
        params['seq_len'] = int(input("请输入序列长度(默认5): ") or 5)
        params['max_bert_len'] = int(input("请输入单本书名最大分词长度(默认10): ") or 10)
        params['hidden_dim'] = int(input("请输入融合层隐藏维度(默认256): ") or 256)
        params['dropout'] = float(input("请输入dropout比例(默认0.3): ") or 0.3)
        params['batch_size'] = int(input("请输入批次大小(默认16，BERT建议 smaller): ") or 16)
        params['lr'] = float(input("请输入学习率(默认2e-5，BERT建议较小): ") or 2e-5)
        params['epochs'] = int(input("请输入训练轮数(默认10): ") or 10)
        params['patience'] = int(input("请输入早停耐心值(默认3): ") or 3)
        params['freeze_bert'] = input("是否冻结BERT权重(1=是,0=否，小数据建议冻结): ") == "1"
        return params

    def load_bert_model(self, model_path):
        """加载BERT模型（与LSTM加载逻辑类似，适配BERT参数）"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))

        # 恢复编码器
        self.bert_user_encoder = checkpoint.get("user_encoder")
        self.bert_book_encoder = checkpoint.get("book_encoder")
        if self.bert_user_encoder is None or self.bert_book_encoder is None:
            raise ValueError("模型文件中未找到用户/图书编码器")

        # 恢复模型超参数
        self.bert_seq_len = checkpoint.get("seq_len", 5)
        self.bert_max_len = checkpoint.get("max_bert_len", 10)
        embedding_dim = checkpoint.get("embedding_dim", 128)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_layers = checkpoint.get("num_layers", 2)
        dropout = checkpoint.get("dropout", 0.3)
        title_vector_dim = checkpoint.get("title_vector_dim", 0)
        book_num = len(self.bert_book_encoder.classes_)

        # 初始化BERT模型
        self.bert_model = BertSeqRecModel(
            book_num=book_num,
            feature_dim=3,
            title_vec_dim=title_vector_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)
        self.bert_model.load_state_dict(checkpoint["model_state_dict"])
        self.bert_model.eval()

        # 加载BERT分词器
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.logger.info(f"BERT模型加载成功: {model_path}")

    def predict_bert_test(self, submission_path="submission_bert.csv"):
        """BERT模型预测测试集（参考LSTM预测逻辑，适配BERT输入）"""
        if self.bert_model is None or self.test_data is None:
            raise ValueError("请先加载BERT模型和测试数据")

        submission_data = []
        pred_list = []
        true_list = []
        self.test_data = pd.DataFrame(self.test_data)
        # 按用户分组处理
        grouped = self.test_data.groupby("user_id")
        for user_id, group in tqdm(grouped, desc="BERT预测测试集用户"):
            sorted_group = group.sort_values("借阅时间").reset_index(drop=True)
            user_interactions = sorted_group.to_dict("records")

            # 调用BERT预测方法（需实现_bert_predict）
            pred_book = self._bert_predict(
                self.bert_model,
                user_interactions,
                self.bert_user_encoder,
                self.bert_book_encoder,
                self.bert_seq_len,
                self.bert_tokenizer,
                self.title_vector_path
            )

            if pred_book is not None:
                submission_data.append({"user_id": user_id, "book_id": pred_book})
                pred_list.append(pred_book)
                true_book = sorted_group["book_id"].iloc[-1] if not sorted_group.empty else None
                true_list.append(true_book)

        # 生成提交文件和评估指标（与LSTM逻辑一致）
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(submission_path, index=False, encoding="utf-8")
        self.logger.info(f"BERT提交文件已保存至: {submission_path}")

        if len(pred_list) > 0 and len(true_list) > 0:
            precision, recall = calculate_precision_recall(pred_list, true_list)
            f1 = calculate_f1(precision, recall)
            print(f"BERT测试集指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

    def _bert_predict(self, model, user_inter, user_encoder, book_encoder, seq_len, tokenizer, title_vector_path):
        """BERT模型单用户预测（适配BERT的输入格式）"""
        try:
            # 提取序列数据
            book_ids = [inter['book_id'] for inter in user_inter]
            features = np.array([
                [inter['交互权重'], inter['dept_book_preference'], inter['dept_cat1_preference']]
                for inter in user_inter
            ], dtype=np.float32)
            titles = [inter.get('书名', '') for inter in user_inter]

            # 加载词向量
            vec_df = pd.read_csv(title_vector_path)
            vector_cols = [col for col in vec_df.columns if col.startswith('title_vec_')]
            title_vec_dict = {str(row['book_id']): row[vector_cols].values for _, row in vec_df.iterrows()}
            title_vec_dim = len(vector_cols) if vector_cols else 0
            title_vecs = np.array([
                title_vec_dict.get(str(book_id), np.zeros(title_vec_dim)) for book_id in book_ids
            ], dtype=np.float32)

            # 截取最近的seq_len条记录
            if len(book_ids) < seq_len:
                return None  # 序列长度不足，返回空
            input_books = book_ids[-seq_len:]
            input_features = features[-seq_len:]
            input_titles = titles[-seq_len:]
            input_title_vecs = title_vecs[-seq_len:]

            # BERT分词
            bert_inputs = tokenizer(
                input_titles,
                padding='max_length',
                truncation=True,
                max_length=self.bert_max_len,
                return_tensors='pt'
            )
            input_ids = bert_inputs['input_ids'].unsqueeze(0)  # [1, seq_len, max_bert_len]
            attention_mask = bert_inputs['attention_mask'].unsqueeze(0)  # [1, seq_len, max_bert_len]

            # 转换为张量并预测
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            title_vecs_tensor = torch.FloatTensor(input_title_vecs).unsqueeze(0).to(
                self.device)  # [1, seq_len, vec_dim]
            features_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)  # [1, seq_len, 3]

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, title_vecs_tensor, features_tensor)
                pred_idx = torch.argmax(outputs, dim=1).cpu().item()

            return book_encoder.inverse_transform([pred_idx])[0]
        except Exception as e:
            self.logger.error(f"BERT预测错误: {str(e)}")
            return None
    def train(self, model_type):
        """扩展支持bert模型的训练入口"""
        if self.train_data is None or self.valid_data is None:
            raise ValueError("请先调用load_and_preprocess_data加载数据")

        print(f"\n===== 配置{model_type}模型训练参数 =====")

        # 根据模型类型选择对应的参数输入函数
        if model_type == 'bert':
            params = self._get_bert_params()  # 调用bert专用参数输入
        else:
            params = self._get_user_input(model_type)  # 原有模型参数输入

        if params['k_fold']:
            return self.cross_validate(model_type, params)
        else:
            if model_type == 'user_cf':
                return self.train_user_cf(params)
            elif model_type == 'lstm':
                return self.train_lstm(params)
            elif model_type == 'hybrid':
                return self.train_hybrid(params)
            elif model_type == 'bert':  # 新增bert模型训练分支
                return self.train_bert(params)
            else:
                raise ValueError("支持的模型类型: 'user_cf', 'lstm', 'hybrid', 'bert'")  # 更新支持列表
    def predict_lstm_test(self, submission_path="submission_lstm.csv", confidence_threshold=0.5):
        """
        修改：添加置信度阈值判断
        - 若模型预测置信度 ≥ threshold：使用模型推荐结果
        - 若置信度 < threshold 或 预测失败：默认推荐用户最后一次借阅的书籍
        :param confidence_threshold: 置信度阈值（0-1之间，默认0.5，可根据需求调整）
        """
        if self.lstm_model is None or self.test_data is None:
            raise ValueError("请先加载LSTM模型和测试数据（调用load_lstm_model和load_and_preprocess_data）")

        submission_data = []
        pred_list = []
        true_list = []
        fallback_count = 0  # 统计触发"默认推荐最后一本书"的次数

        # 按用户分组并按时间排序（确保时序正确）
        grouped = self.test_data.groupby("user_id")
        for user_id, group in tqdm(grouped, desc="预测测试集用户"):
            sorted_group = group.sort_values("借阅时间").reset_index(drop=True)
            user_interactions = sorted_group.to_dict("records")
            # 获取用户最后一次借阅的书籍（作为默认推荐）
            user_last_book = sorted_group["book_id"].iloc[-1] if not sorted_group.empty else None

            # 调用修改后的预测方法，获取（预测书籍ID，置信度）
            pred_book, pred_confidence = self._lstm_predict(
                self.lstm_model,
                user_interactions,
                self.lstm_user_encoder,
                self.lstm_book_encoder,
                self.lstm_seq_len,
                title_vector_path=self.title_vector_path
            )

            # 核心判断逻辑
            if pred_book is not None and pred_confidence >= confidence_threshold:
                # 置信度达标：使用模型推荐结果
                final_book = pred_book
            else:
                # 置信度不达标或预测失败：使用默认推荐（最后一次借阅的书）
                final_book = user_last_book
                fallback_count += 1

            # 收集结果
            if final_book is not None:
                submission_data.append({"user_id": user_id, "book_id": final_book})
                pred_list.append(final_book)
                # 真实标签：用户最后一次借阅的书籍（任务场景默认标签）
                true_book = user_last_book
                true_list.append(true_book)

        # 生成提交文件
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(submission_path, index=False, encoding="utf-8")
        self.logger.info(f"提交文件已保存至: {submission_path}，共 {len(submission_df)} 条记录")
        self.logger.info(f"触发默认推荐（最后一本书）的次数: {fallback_count} / {len(grouped)} 个用户")
        print(f"提交文件已保存至: {submission_path}，共 {len(submission_df)} 条记录")
        print(f"触发默认推荐（最后一本书）的次数: {fallback_count} / {len(grouped)} 个用户")

        # 计算评估指标
        if len(pred_list) > 0 and len(true_list) > 0:
            precision, recall = calculate_precision_recall(pred_list, true_list)
            f1 = calculate_f1(precision, recall)
            self.logger.info(f"测试集评估指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
            print(f"测试集评估指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        else:
            self.logger.warning("无有效预测结果，无法计算评估指标")
            print("无有效预测结果，无法计算评估指标")

    # -------------------------- 原有交叉验证和训练入口 --------------------------
    def cross_validate(self, model_type, params):
        """原有交叉验证逻辑（保持不变）"""
        n_splits = params['n_splits']
        print(f"\n=== 开始{model_type}模型的{n_splits}折交叉验证 ===")
        self.logger.info(f"开始{model_type}的{n_splits}折交叉验证")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        total_start = time.time()

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.train_data)):
            fold_start = time.time()
            print(f"\n===== 第{fold + 1}/{n_splits}折 =====")
            self.logger.info(f"第{fold + 1}折交叉验证开始")

            original_train = self.train_data
            original_valid = self.valid_data
            self.train_data = original_train.iloc[train_idx].copy()
            self.valid_data = original_train.iloc[valid_idx].copy()

            if model_type == 'user_cf':
                _, f1 = self.train_user_cf(params)
            elif model_type == 'lstm':
                _, _, _, f1 = self.train_lstm(params)
            elif model_type == 'hybrid':
                _, f1 = self.train_hybrid(params)

            fold_scores.append(f1)
            fold_time = time.time() - fold_start
            print(f"第{fold + 1}折完成 - F1: {f1:.4f}, 用时: {fold_time:.2f}秒")
            self.logger.info(f"第{fold + 1}折完成 - F1: {f1:.4f}")

            self.train_data = original_train
            self.valid_data = original_valid

        avg_score = np.mean(fold_scores)
        total_time = time.time() - total_start
        print(
            f"\n{n_splits}折交叉验证完成 - 平均F1: {avg_score:.4f} ± {np.std(fold_scores):.4f}, 总耗时: {total_time:.2f}秒")
        self.logger.info(f"{n_splits}折交叉验证平均F1: {avg_score:.4f}")
        return avg_score



def main():
    # 1. 定义数据路径
    data_paths = {
        "books": "./data/item.csv",
        "borrows": "./data/inter_final_选手可见.csv",
        "users": "./data/user.csv"
    }

    # 2. 初始化训练器
    trainer = EnhancedModelTrainer(data_paths)

    # 3. 加载并预处理数据（包含测试集）
    # 可设置train_word2vec=False跳过Word2Vec训练（已训练过模型时）
    trainer.load_and_preprocess_data(
        test_size=0.2,
        random_state=42,
        train_word2vec=False  # 首次运行设为True，后续可设为False
    )

    # 4. 选择操作：训练模型 或 直接预测
    action = input("请选择操作 (1=训练LSTM模型, 2=加载LSTM模型并预测, 3=训练BERT模型, 4=加载BERT模型并预测): ")
    if action == "1":
        # 训练LSTM模型
        trainer.train("lstm")
    elif action == "2":
        # 加载LSTM模型并预测
        default_lstm_path = "model/saved_models/lstm_best_epoch167_f10.0049.pth"
        model_path = input(f"请输入LSTM模型路径（默认: {default_lstm_path}）: ") or default_lstm_path
        try:
            trainer.load_lstm_model(model_path)
            trainer.predict_lstm_test(confidence_threshold=0.5)
        except Exception as e:
            print(f"LSTM预测失败: {str(e)}")
    elif action == "3":
        # 训练BERT模型
        trainer.train("bert")
    elif action == "4":
        # 加载BERT模型并预测（需先实现load_bert_model和predict_bert_test方法）
        default_bert_path = "model/saved_models/bert_best_epoch10_f10.0085.pth"  # 示例默认路径
        model_path = input(f"请输入BERT模型路径（默认: {default_bert_path}）: ") or default_bert_path
        try:
            trainer.load_bert_model(model_path)  # 需补充实现BERT模型加载方法
            trainer.predict_bert_test()  # 需补充实现BERT预测方法
        except Exception as e:
            print(f"BERT预测失败: {str(e)}")
    else:
        print("无效操作，请输入1-4")


if __name__ == "__main__":
    main()