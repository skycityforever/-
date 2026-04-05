import pandas as pd
import torch
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import random

# 数据预处理模块导入
from 图书馆.library_book_recommendation.data_preprocessing.data_cleaning import clean_book_secondary_category, clean_inter_data  # 清洗函数
from 图书馆.library_book_recommendation.data_preprocessing.data_splitting import merge_and_split_data  # 数据合并函数
from 图书馆.library_book_recommendation.data_preprocessing.feature_engineering import (
    extract_time_features,
    build_interaction_weight,
    encode_categorical_features,
    merge_dept_features_to_inter,
    add_user_dept_features
)

# 模型和工具导入
from 图书馆.library_book_recommendation.model.base_models.user_cf import UserCF
from 图书馆.library_book_recommendation.model.advanced_models.lstm_seq_rec import LSTMSeqRecModel, BorrowSequenceDataset
from 图书馆.library_book_recommendation.utils.logger import Logger
from 图书馆.library_book_recommendation.evaluation.metrics import calculate_precision_recall, calculate_f1


class EnhancedModelTrainer:
    def __init__(self, data_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        :param data_paths: 数据路径字典，包含'books'、'borrows'等路径
        :param device: 训练设备
        """
        # 设备配置与日志
        self.test_data = None
        self.book_cat_encoder = None
        self.user_encoder = None
        self.device = self._init_device(device)
        self.logger = Logger('enhanced_training.log').get_logger()
        self.logger.info(f"训练设备初始化完成: {self.device}")

        # 数据路径与预处理结果
        self.data_paths = data_paths
        self.cleaned_data = None  # 清洗后的数据
        self.train_data = None  # 训练集
        self.valid_data = None  # 验证集

        # 训练记录
        self.loss_history = {}
        self.metric_history = {}
        self.best_models = {}  # 存储各模型最佳版本
        self.best_metrics = {}  # 存储各模型最佳指标

    def _init_device(self, device):
        """初始化设备，优先使用CUDA"""
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，自动切换到CPU")
            return 'cpu'
        return device

    # -------------------------- 数据预处理流程 --------------------------
    def load_and_preprocess_data(self, test_size=0.2, random_state=42):
        """
        完整数据预处理流程：清洗原始数据 -> 保存清洗后文件 -> 调用merge_and_split_data合并分割
        """
        self.logger.info("开始数据预处理...")
        start_time = time.time()

        # ---------------------- 1. 清洗原始数据并保存为临时文件 ----------------------
        # 清洗图书数据并保存
        books_clean = clean_book_secondary_category(self.data_paths['books'])  # 调用清洗函数
        cleaned_books_path = "data/cleaned_books.csv"  # 清洗后图书数据路径
        books_clean.to_csv(cleaned_books_path, index=False, encoding='utf-8')
        self.logger.info(f"图书数据清洗完成并保存: {len(books_clean)}条 -> {cleaned_books_path}")

        # 清洗借阅数据并保存
        borrows_clean = clean_inter_data(self.data_paths['borrows'])  # 调用清洗函数
        cleaned_borrows_path = "data/cleaned_borrows.csv"  # 清洗后借阅数据路径
        borrows_clean.to_csv(cleaned_borrows_path, index=False, encoding='utf-8')
        self.logger.info(f"借阅数据清洗完成并保存: {len(borrows_clean)}条 -> {cleaned_borrows_path}")

        # ---------------------- 2. 调用merge_and_split_data合并分割（传文件路径） ----------------------
        # 定义合并/分割后的数据保存路径
        merged_path = "data/merged_data.csv"
        train_path = "data/train_data.csv"
        valid_path = "data/valid_data.csv"
        test_path = "data/test_data.csv"

        # 调用data_splitting.py中的函数（参数完全匹配其要求）
        merge_and_split_data(
            inter_path=cleaned_borrows_path,  # 清洗后的借阅数据路径（替代原始交互数据）
            item_path=cleaned_books_path,  # 清洗后的图书数据路径（替代原始图书数据）
            user_path=self.data_paths['users'],  # 用户数据原始路径（假设无需提前清洗）
            merged_path=merged_path,
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            encoding='utf-8',  # 编码与保存时一致
            split_ratios=(1 - test_size - 0.1, 0.1, test_size),  # 训练集:验证集:测试集（总和1.0）
            random_state=random_state
        )

        # ---------------------- 3. 训练集执行特征工程 ----------------------
        # 读取训练集的数据
        train_data = pd.read_csv(train_path, encoding='utf-8',
                                    parse_dates=["借阅时间", "还书时间", "续借时间"])
        train_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        train_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')  # 读取原始用户数据

        # ① 时间特征（借阅月份、是否学期内）
        train_inter_enhanced = extract_time_features(train_data.copy())
        # ② 交互权重（续借次数→兴趣强度）
        train_inter_enhanced = build_interaction_weight(train_inter_enhanced)
        # ③ 分类特征编码（用户院系、图书分类）
        train_users_encoded, train_books_encoded = encode_categorical_features(train_users.copy(), train_books.copy())
        # 下面两行做一下测试
        self.user_encoder = train_users_encoded['DEPT'].astype('category').cat.categories
        self.book_cat_encoder = train_books_encoded['二级分类'].astype('category').cat.categories
        # ④ 合并院系-图书/分类偏好到交互表
        train_inter_enhanced = merge_dept_features_to_inter(train_inter_enhanced.copy(), train_users.copy(),
                                                            train_books.copy())
        # ⑤ 增强用户表（院系平均续借、分类偏好等）
        train_users_enhanced = add_user_dept_features(train_users_encoded.copy(), train_inter_enhanced.copy(),
                                                      train_books.copy())

        # 保存增强后的数据（可选，用于 debug 或后续复用）
        train_inter_enhanced.to_csv("data/cleaned_borrows_enhanced.csv", index=False, encoding='utf-8')
        train_books_encoded.to_csv("data/cleaned_books_encoded.csv", index=False, encoding='utf-8')
        train_users_enhanced.to_csv("data/cleaned_user_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"训练集特征工程完成: {len(train_inter_enhanced)}条 -> cleaned_borrows_enhanced.csv")

        # ---------------------- 4. 验证集特征工程（复用训练集编码/统计逻辑） ----------------------
        valid_data = pd.read_csv(valid_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        valid_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        valid_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')

        # 时间特征提取（同训练集逻辑）
        valid_inter_enhanced = extract_time_features(valid_data.copy())
        # 交互权重构建（同训练集逻辑）
        valid_inter_enhanced = build_interaction_weight(valid_inter_enhanced)
        # 分类特征编码（使用训练集编码器）
        valid_users_encoded = valid_users.copy().rename(columns={"借阅人": "user_id"})
        valid_users_encoded['DEPT_encoded'] = valid_users_encoded['DEPT'].astype('category').cat.set_categories(
            self.user_encoder).cat.codes
        valid_books_encoded = pd.get_dummies(valid_books, columns=['一级分类'], prefix='book_cat1')
        valid_books_encoded['二级分类_encoded'] = valid_books_encoded['二级分类'].astype('category').cat.set_categories(
            self.book_cat_encoder).cat.codes
        # 合并院系特征（同训练集逻辑）
        valid_inter_enhanced = merge_dept_features_to_inter(valid_inter_enhanced.copy(), valid_users.copy(),
                                                            valid_books.copy())
        # 增强用户表（复用训练集统计结果）
        valid_users_enhanced = add_user_dept_features(valid_users_encoded.copy(), valid_inter_enhanced.copy(),
                                                      valid_books.copy())

        # 保存验证集增强结果（可选，若需复用）
        valid_inter_enhanced.to_csv("data/valid_borrows_enhanced.csv", index=False, encoding='utf-8')
        valid_books_encoded.to_csv("data/valid_books_encoded.csv", index=False, encoding='utf-8')
        valid_users_enhanced.to_csv("data/valid_user_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"验证集特征处理完成: {len(valid_inter_enhanced)}条 -> valid_borrows_enhanced.csv")

        # ---------------------- 5. 测试集特征工程（复用训练集编码/统计逻辑） ----------------------
        # 这里只是保证代码完整性做的测试，后面会写一个新的文件去预测测试集数据并计算F1分数
        test_data = pd.read_csv(test_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
        test_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
        test_users = pd.read_csv(self.data_paths['users'], encoding='utf-8')

        # 时间特征提取（同训练集逻辑）
        test_inter_enhanced = extract_time_features(test_data.copy())
        # 交互权重构建（同训练集逻辑）
        test_inter_enhanced = build_interaction_weight(test_inter_enhanced)
        # 分类特征编码（使用训练集编码器）
        test_users_encoded = test_users.copy().rename(columns={"借阅人": "user_id"})
        test_users_encoded['DEPT_encoded'] = test_users_encoded['DEPT'].astype('category').cat.set_categories(
            self.user_encoder).cat.codes
        test_books_encoded = pd.get_dummies(test_books, columns=['一级分类'], prefix='book_cat1')
        test_books_encoded['二级分类_encoded'] = test_books_encoded['二级分类'].astype('category').cat.set_categories(
            self.book_cat_encoder).cat.codes
        # 合并院系特征（同训练集逻辑）
        test_inter_enhanced = merge_dept_features_to_inter(test_inter_enhanced.copy(), test_users.copy(),
                                                            test_books.copy())
        # 增强用户表（复用训练集统计结果）
        test_users_enhanced = add_user_dept_features(test_users_encoded.copy(), test_inter_enhanced.copy(),
                                                      test_books.copy())

        # 保存测试集增强结果（可选，若需复用）
        test_inter_enhanced.to_csv("data/test_borrows_enhanced.csv", index=False, encoding='utf-8')
        test_books_encoded.to_csv("data/test_books_encoded.csv", index=False, encoding='utf-8')
        test_users_enhanced.to_csv("data/test_user_enhanced.csv", index=False, encoding='utf-8')
        self.logger.info(f"验证集特征处理完成: {len(valid_inter_enhanced)}条 -> test_borrows_enhanced.csv")



        # ---------------------- 5. 加载分割后的训练集和验证集 ----------------------
        self.train_data = pd.read_csv("data/cleaned_borrows_enhanced.csv", encoding='utf-8')
        self.valid_data = pd.read_csv("data/valid_borrows_enhanced.csv", encoding='utf-8')
        self.test_data = pd.read_csv("data/test_borrows_enhanced.csv", encoding='utf-8')
        self.logger.info(f"数据分割完成 - 训练集: {len(self.train_data)}条, 验证集: {len(self.valid_data)}条")
        self.logger.info(f"数据预处理总耗时: {time.time() - start_time:.2f}秒")

    # -------------------------- 参数配置 --------------------------
    def _get_user_input(self, model_type):
        """获取用户输入的超参数和训练配置"""
        params = {}

        # 通用参数
        params['k_fold'] = int(input("是否使用K折交叉验证？(1=是,0=否): "))
        if params['k_fold']:
            params['n_splits'] = int(input("请输入折数(默认5): ") or 5)

        # 模型特定参数
        if model_type == 'lstm':
            default_epochs = 30
            params['epochs'] = int(input(f"请输入训练轮数(默认{default_epochs}): ") or default_epochs)

        # 模型超参数
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
            params['patience'] = int(input("请输入早停耐心值(默认3): ") or 3)  # 早停机制
        elif model_type == 'hybrid':
            params['cf_weight'] = float(input("请输入协同过滤权重(0-1,默认0.5): ") or 0.5)
            params['lstm_weight'] = 1 - params['cf_weight']
            print("=== 请配置协同过滤子模型参数 ===")
            params['cf_params'] = self._get_user_input('user_cf')
            print("=== 请配置LSTM子模型参数 ===")
            params['lstm_params'] = self._get_user_input('lstm')

        return params

    # -------------------------- 训练可视化 --------------------------
    def _visualize_step(self, model_type, epoch, loss, acc, f1, epoch_time):
        """实时可视化单轮训练情况"""
        print(f"\rEpoch {epoch:3d} - "
              f"损失: {loss:.4f} | "
              f"准确率: {acc:.4f} | "
              f"F1分数: {f1:.4f} | "
              f"用时: {epoch_time:.2f}秒", end='')

    def _visualize_summary(self, model_type):
        """训练结束后可视化完整曲线"""
        if model_type not in self.loss_history:
            print("无训练历史可可视化")
            return

        print("\n" + "=" * 60)
        print(f"{model_type}模型训练完整记录")
        print("=" * 60)

        # 损失曲线
        loss_data = self.loss_history[model_type]
        max_loss, min_loss = max(loss_data), min(loss_data)
        print("\n损失变化趋势:")
        for i, loss in enumerate(loss_data):
            bar_len = int(30 * (1 - (loss - min_loss) / (max_loss - min_loss + 1e-8)))
            print(f"Epoch {i + 1:3d}: {loss:.4f} " + "|" + "*" * bar_len + " " * (30 - bar_len) + "|")

        # F1曲线
        if model_type in self.metric_history:
            f1_data = [m['f1'] for m in self.metric_history[model_type]]
            print("\nF1分数变化趋势:")
            for i, f1 in enumerate(f1_data):
                bar_len = int(30 * f1)
                print(f"Epoch {i + 1:3d}: {f1:.4f} " + "|" + "*" * bar_len + " " * (30 - bar_len) + "|")

    # -------------------------- 模型训练 --------------------------
    def train_user_cf(self, params):
        """训练用户协同过滤模型（适配修改后的UserCF类）"""
        model_type = 'user_cf'
        start_time = time.time()
        print("\n=== 开始训练用户协同过滤模型 ===")
        self.logger.info("开始训练用户协同过滤模型")

        # 1. 初始化模型（top_k从参数获取）
        model = UserCF(top_k=params['top_k'])

        # 2. 调用fit方法统一训练（内部自动构建矩阵、计算相似度，并保存user_idx等属性）
        model.fit(self.train_data)  # 传入训练数据，fit方法会处理所有初始化逻辑

        # 3. 评估（此时model已包含user_idx、book_idx等属性，无需额外传入user_list和book_list）
        precision, recall = calculate_precision_recall(model, self.valid_data)  # 简化评估函数参数
        f1_score = calculate_f1(precision, recall)
        acc = precision  # 协同过滤用精确率近似准确率

        # 4. 记录指标
        self.loss_history[model_type] = [0.0]  # CF无损失概念，用0占位
        self.metric_history[model_type] = [{'precision': precision, 'recall': recall, 'f1': f1_score, 'acc': acc}]

        # 5. 保存最佳模型
        self._save_best_model(model, model_type, 1, f1_score)

        # 6. 输出结果
        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        print(f"验证集指标 - 准确率: {acc:.4f}, F1: {f1_score:.4f}")
        self.logger.info(f"用户协同过滤训练完成 - 准确率: {acc:.4f}, F1: {f1_score:.4f}, 耗时: {total_time:.2f}秒")

        return model, f1_score

    def train_lstm(self, params):
        """训练LSTM序列推荐模型"""
        model_type = 'lstm'
        start_time = time.time()
        print("\n=== 开始训练LSTM模型 ===")
        self.logger.info("开始训练LSTM模型")

        # 数据编码（移至设备）
        user_encoder = LabelEncoder()
        book_encoder = LabelEncoder()
        user_encoder.fit(self.train_data['user_id'].unique())
        all_book_ids = pd.concat([self.train_data['book_id'], self.valid_data['book_id']]).unique()
        book_encoder.fit(all_book_ids)
        book_num = len(book_encoder.classes_)

        # 创建数据集和数据加载器
        train_dataset = BorrowSequenceDataset(
            self.train_data, user_encoder, book_encoder, seq_len=params['seq_len']
        )
        valid_dataset = BorrowSequenceDataset(
            self.valid_data, user_encoder, book_encoder, seq_len=params['seq_len']
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])

        # 初始化模型（移至设备）
        model = LSTMSeqRecModel(
            book_num=book_num,
            feature_dim=3,
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        self.logger.info(f"LSTM模型已加载至设备: {self.device}")

        # 损失函数与优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # 训练记录初始化
        self.loss_history[model_type] = []
        self.metric_history[model_type] = []
        best_f1 = 0.0
        no_improve_epochs = 0

        # 训练循环
        for epoch in range(1, params['epochs'] + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            train_preds = []
            train_trues = []

            # 训练轮次
            for input_ids, input_features, target in tqdm(train_loader, desc=f"Epoch {epoch}/{params['epochs']}", leave=False):
                # 数据移至设备
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

            # 计算训练指标
            avg_loss = total_loss / len(train_loader)
            # 这里忘了评估软件包也写了
            train_precision, train_recall = self._calculate_sequence_metrics(train_preds, train_trues)
            train_f1 = calculate_f1(train_precision, train_recall)
            train_acc = train_precision  # 序列预测准确率用精确率近似

            # 验证评估
            model.eval()
            with torch.no_grad():
                valid_preds = []
                valid_trues = []
                for input_ids, input_features, target in valid_loader:
                    input_ids = input_ids.to(self.device)
                    input_features = input_features.to(self.device)  # 特征移至设备
                    outputs = model(input_ids, input_features)  # 传入模型
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    valid_preds.extend(preds)
                    valid_trues.extend(target.cpu().numpy())

            valid_precision, valid_recall = self._calculate_sequence_metrics(valid_preds, valid_trues)
            valid_f1 = calculate_f1(valid_precision, valid_recall)
            valid_acc = valid_precision

            # 记录历史
            self.loss_history[model_type].append(avg_loss)
            self.metric_history[model_type].append({
                'train_acc': train_acc, 'train_f1': train_f1,
                'valid_acc': valid_acc, 'valid_f1': valid_f1,
                'epoch': epoch
            })

            # 实时可视化
            epoch_time = time.time() - epoch_start
            self._visualize_step(model_type, epoch, avg_loss, valid_acc, valid_f1, epoch_time)

            # 保存最佳模型（基于验证集F1）
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                self._save_best_model(model, model_type, epoch, valid_f1,
                                      user_encoder=user_encoder, book_encoder=book_encoder)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= params['patience']:
                    self.logger.info(f"早停触发: 连续{params['patience']}轮无提升")
                    break

        # 训练总结
        total_time = time.time() - start_time
        # self._visualize_summary(model_type)
        print(f"\n训练完成! 总耗时: {total_time:.2f}秒")
        print(f"最佳验证集F1: {best_f1:.4f}")
        self.logger.info(f"LSTM训练完成 - 最佳F1: {best_f1:.4f}, 总耗时: {total_time:.2f}秒")

        return model, user_encoder, book_encoder, best_f1

    def train_hybrid(self, params):
        """训练混合模型（协同过滤 + LSTM）"""
        model_type = 'hybrid'
        start_time = time.time()
        print("\n=== 开始训练混合模型 ===")
        self.logger.info("开始训练混合模型")

        # 训练子模型
        cf_model, cf_f1 = self.train_user_cf(params['cf_params'])
        lstm_model, user_encoder, book_encoder, lstm_f1 = self.train_lstm(params['lstm_params'])

        # 混合模型封装
        class HybridModel:
            def __init__(self, cf_model, lstm_model, user_encoder, book_encoder,
                         cf_weight=0.5, lstm_weight=0.5, seq_len=5, trainer=None):
                self.cf_model = cf_model
                self.lstm_model = lstm_model
                self.user_encoder = user_encoder
                self.book_encoder = book_encoder
                self.cf_weight = cf_weight
                self.lstm_weight = lstm_weight
                self.seq_len = seq_len
                self.trainer = trainer  # 引用训练器的预测方法

            def predict(self, user_id, user_idx, book_idx, book_list, user_inter=None):
                # CF推荐
                cf_rec = self.cf_model.predict(user_id, user_idx, book_idx, book_list)

                # LSTM推荐（使用训练器的预测逻辑）
                if user_inter is not None and len(user_inter) >= self.seq_len:
                    lstm_rec = self.trainer._lstm_predict(
                        self.lstm_model, user_inter, self.user_encoder,
                        self.book_encoder, self.seq_len
                    )
                    lstm_rec = lstm_rec if lstm_rec is not None else cf_rec
                else:
                    lstm_rec = cf_rec

                # 加权融合
                return cf_rec if random.random() < self.cf_weight else lstm_rec

        # 初始化混合模型
        hybrid_model = HybridModel(
            cf_model, lstm_model, user_encoder, book_encoder,
            params['cf_weight'], params['lstm_weight'],
            params['lstm_params']['seq_len'], self
        )

        # 评估混合模型
        user_list = self.train_data['user_id'].unique().tolist()
        book_list = self.train_data['book_id'].unique().tolist()
        user_idx = {u: i for i, u in enumerate(user_list)}
        book_idx = {b: i for i, b in enumerate(book_list)}

        precision, recall = calculate_precision_recall(
            hybrid_model, self.valid_data, user_list, book_list
        )
        f1_score = calculate_f1(precision, recall)
        acc = precision

        # 保存最佳模型
        self._save_best_model(hybrid_model, model_type, 1, f1_score)

        # 输出结果
        total_time = time.time() - start_time
        print(f"\n混合模型训练完成! 总耗时: {total_time:.2f}秒")
        print(f"验证集指标 - 准确率: {acc:.4f}, F1: {f1_score:.4f}")
        self.logger.info(f"混合模型训练完成 - 准确率: {acc:.4f}, F1: {f1_score:.4f}, 耗时: {total_time:.2f}秒")

        return hybrid_model, f1_score

    # -------------------------- 辅助函数 --------------------------
    def calculate_precision_recall(model, valid_data):
        """
        计算精确率和召回率（适配修改后的UserCF，无需外部传入user_list和book_list）
        参数：
            model: 训练好的UserCF模型（内部含user_idx、book_list等属性）
            valid_data: 验证集数据（DataFrame，含user_id、book_id等）
        返回：
            precision: 精确率
            recall: 召回率
        """
        hits = 0  # 预测正确的数量
        total_pred = 0  # 总预测次数（每个用户预测1本）
        total_true = 0  # 总真实借阅数量（验证集中用户实际借阅的图书数）

        # 按用户分组评估（从验证集中取用户，避免评估训练集中不存在的用户）
        for user_id, user_group in valid_data.groupby('user_id'):
            # 跳过训练集中不存在的用户（新用户，模型无法预测）
            if user_id not in model.user_idx:
                continue

            # 1. 获取当前用户的真实借阅图书（验证集中的实际记录）
            true_books = set(user_group['book_id'].tolist())
            if not true_books:  # 无真实借阅记录，跳过
                continue
            total_true += len(true_books)  # 累计真实借阅数

            # 2. 模型预测（调用修改后的predict方法，无需传额外参数）
            pred_book = model.predict(user_id)
            total_pred += 1  # 累计预测次数

            # 3. 判断预测是否正确（预测的图书在真实借阅列表中）
            if pred_book in true_books:
                hits += 1  # 累计预测正确数

        # 避免除零错误（若无有效预测或无真实数据）
        precision = hits / total_pred if total_pred > 0 else 0.0
        recall = hits / total_true if total_true > 0 else 0.0

        return precision, recall
    def _calculate_sequence_metrics(self, preds, trues):
        """计算序列推荐的精确率和召回率"""
        tp = sum(p == t for p, t in zip(preds, trues))
        fp = len(preds) - tp
        fn = len(trues) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def _lstm_predict(self, model, user_inter, user_encoder, book_encoder, seq_len):
        """LSTM预测逻辑（适配新增特征的模型）"""
        try:
            # 1. 提取图书ID和对应的特征（从user_inter中获取）
            # 假设user_inter是包含所有字段的列表（如[{book_id:..., 交互权重:..., dept_book_preference:..., ...}, ...]）
            book_ids = [inter['book_id'] for inter in user_inter]
            # 提取新增特征（根据实际特征名称调整字段名）
            features = [
                [inter['交互权重'], inter['dept_book_preference'], inter['dept_cat1_preference']]
                for inter in user_inter
            ]

            # 2. 编码图书ID
            encoded_books = book_encoder.transform(book_ids)
            # 截取最近的seq_len个（和特征保持长度一致）
            input_ids = encoded_books[-seq_len:].astype(np.int64)
            input_features = np.array(features[-seq_len:], dtype=np.float32)  # 特征同样截取最近的seq_len个

            # 3. 序列长度不足时补全（同步补全ID和特征）
            if len(input_ids) < seq_len:
                # 补全图书ID（用0填充，假设0是无效编码）
                pad_length = seq_len - len(input_ids)
                input_ids = np.pad(input_ids, (pad_length, 0), mode='constant')
                # 补全特征（用0填充，或用训练集特征均值）
                input_features = np.pad(input_features, ((pad_length, 0), (0, 0)), mode='constant')

            # 4. 转为张量并增加批次维度（模型要求输入为[batch_size, seq_len, ...]）
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(
                self.device)  # shape: [1, seq_len]
            input_features_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(
                self.device)  # shape: [1, seq_len, 3]

            # 5. 模型预测（同时传入ID和特征）
            model.eval()
            with torch.no_grad():
                output = model(input_ids_tensor, input_features_tensor)  # 适配修改后的模型输入
                pred_idx = torch.argmax(output, dim=1).cpu().item()

            # 6. 解码并返回预测结果
            return book_encoder.inverse_transform([pred_idx])[0]
        except Exception as e:
            self.logger.error(f"LSTM预测错误: {str(e)}")
            return None

    def _save_best_model(self, model, model_type, epoch, f1_score, **kwargs):
        """保存最佳模型（兼容LSTM等PyTorch模型 + UserCF等自定义模型）"""
        # 1. 创建保存目录（沿用原始路径）
        save_dir = "model/saved_models"
        os.makedirs(save_dir, exist_ok=True)

        # 2. 模型文件名（保持原始命名规则）
        save_path = f"{save_dir}/{model_type}_best_epoch{epoch}_f1{f1_score:.4f}.pth"

        # 3. 区分模型类型，构建保存数据
        save_data = {"f1_score": f1_score, "epoch": epoch}  # 所有模型共用的基础信息
        save_data.update(kwargs)  # 保留原始逻辑：传递额外参数（如LSTM的编码器）

        if isinstance(model, torch.nn.Module):
            # 3.1 PyTorch模型（如LSTM）：保存state_dict（原始逻辑）
            save_data["model_state_dict"] = model.state_dict()
        else:
            # 3.2 自定义模型（如UserCF）：保存核心属性（无state_dict）
            # 需确保UserCF实例已包含这些属性（之前修改的UserCF类已满足）
            save_data.update({
                "user_similarity": model.user_similarity,  # 用户相似度矩阵
                "user_item_matrix": model.user_item_matrix,  # 用户-物品交互矩阵
                "user_idx": model.user_idx,  # 用户ID→索引映射
                "book_idx": model.book_idx,  # 图书ID→索引映射
                "book_list": model.book_list,  # 图书ID列表（索引→ID）
                "top_k": model.top_k  # UserCF的近邻数量参数
            })

        # 4. 保存模型（沿用原始torch.save，支持保存Python对象/NumPy数组）
        torch.save(save_data, save_path)

        # 5. 更新最佳模型记录（原始逻辑不变）
        self.best_models[model_type] = save_path
        self.best_metrics[model_type] = f1_score
        self.logger.info(f"最佳{model_type}模型已保存至: {save_path} (F1: {f1_score:.4f})")

    def cross_validate(self, model_type, params):
        """K折交叉验证"""
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

            # 临时替换训练/验证集
            original_train = self.train_data
            original_valid = self.valid_data
            self.train_data = original_train.iloc[train_idx].copy()
            self.valid_data = original_train.iloc[valid_idx].copy()

            # 训练当前折
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

            # 恢复原始数据
            self.train_data = original_train
            self.valid_data = original_valid

        # 交叉验证总结
        avg_score = np.mean(fold_scores)
        total_time = time.time() - total_start
        print(
            f"\n{n_splits}折交叉验证完成 - 平均F1: {avg_score:.4f} ± {np.std(fold_scores):.4f}, 总耗时: {total_time:.2f}秒")
        self.logger.info(f"{n_splits}折交叉验证平均F1: {avg_score:.4f}")
        return avg_score

    def train(self, model_type):
        """主训练入口"""
        # 检查数据是否已加载
        if self.train_data is None or self.valid_data is None:
            raise ValueError("请先调用load_and_preprocess_data加载数据")

        # 获取用户配置
        print(f"\n===== 配置{model_type}模型训练参数 =====")
        params = self._get_user_input(model_type)

        # 训练逻辑
        if params['k_fold']:
            return self.cross_validate(model_type, params)
        else:
            if model_type == 'user_cf':
                return self.train_user_cf(params)
            elif model_type == 'lstm':
                return self.train_lstm(params)
            elif model_type == 'hybrid':
                return self.train_hybrid(params)
            else:
                raise ValueError("支持的模型类型: 'user_cf', 'lstm', 'hybrid'")

def main():
    # ---------- 1. 定义数据路径 ----------
    data_paths = {
        "books": "./data/item.csv",        # 图书信息表路径
        "borrows": "./data/inter_preliminary.csv",     # 借阅交互表路径
        "users": "./data/user.csv"         # 用户信息表路径（若需清洗用户数据）
    }

    # ---------- 2. 初始化训练器 ----------
    trainer = EnhancedModelTrainer(data_paths)

    # ---------- 3. 加载并预处理数据 ----------
    trainer.load_and_preprocess_data(
        test_size=0.2,    # 验证集比例
        random_state=42   # 随机种子（复现结果）
    )

    # ---------- 4. 选择模型类型并训练 ----------
    # 可选模型类型：'user_cf'（用户协同过滤）、'lstm'（序列推荐）、'hybrid'（混合模型）
    model_type = "lstm"  # 此处以LSTM模型为例，可根据需求修改
    trainer.train(model_type)

if __name__ == "__main__":
    main()