import pandas as pd
import numpy as np
import torch
import time
import os
import random
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# 镜像源设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 数据预处理模块
class DataProcessor:
    def __init__(self, data_paths, logger):
        self.data_paths = data_paths
        self.logger = logger
        self.cleaned_books_path = "data/cleaned_books.csv"
        self.cleaned_borrows_path = "data/cleaned_borrows.csv"
        self.merged_borrows_path = "data/merged_borrows_enhanced.csv"
        self.title_vector_path = "data/item_with_title_vector.csv"

        # 预处理后的数据
        self.books_data = None
        self.borrows_data = None
        self.users_data = None
        self.merged_data = None

        # 编码器
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        self.cat_encoders = {}

    def clean_data(self):
        """清洗数据"""
        self.logger.info("开始数据清洗...")
        start_time = time.time()

        # 清洗图书数据
        books = pd.read_csv(self.data_paths['books'], encoding='utf-8')
        # 处理图书分类
        books['二级分类'] = books['二级分类'].fillna('未知')
        books['一级分类'] = books['一级分类'].fillna('未知')
        books.to_csv(self.cleaned_books_path, index=False, encoding='utf-8')
        self.books_data = books

        # 清洗借阅数据
        borrows = pd.read_csv(self.data_paths['borrows'], encoding='utf-8', parse_dates=['借阅时间', '还书时间'])
        # 计算借阅时长
        borrows['借阅时长'] = (borrows['还书时间'] - borrows['借阅时间']).dt.days
        borrows['借阅时长'] = borrows['借阅时长'].fillna(borrows['借阅时长'].median())
        # 过滤异常值
        borrows = borrows[(borrows['借阅时长'] > 0) & (borrows['借阅时长'] < 365)]
        borrows.to_csv(self.cleaned_borrows_path, index=False, encoding='utf-8')
        self.borrows_data = borrows

        # 加载用户数据
        users = pd.read_csv(self.data_paths['users'], encoding='utf-8')
        users['DEPT'] = users['DEPT'].fillna('未知')
        self.users_data = users

        self.logger.info(f"数据清洗完成，耗时: {time.time() - start_time:.2f}秒")
        self.logger.info(f"图书数据: {len(books)}条，借阅数据: {len(borrows)}条，用户数据: {len(users)}条")

    def feature_engineering(self):
        """特征工程"""
        self.logger.info("开始特征工程...")
        start_time = time.time()

        # 合并数据
        merged = self.borrows_data.merge(self.books_data, on='book_id', how='left')
        merged = merged.merge(self.users_data.rename(columns={'借阅人': 'user_id'}), on='user_id', how='left')

        # 构建交互权重特征
        merged['交互权重'] = 1 + merged['借阅时长'] / 30  # 基础权重+时长影响

        # 构建用户-部门偏好特征
        dept_book_counts = merged.groupby(['DEPT', 'book_id']).size().reset_index(name='count')
        dept_total = merged.groupby('DEPT').size().reset_index(name='total')
        dept_book_counts = dept_book_counts.merge(dept_total, on='DEPT')
        dept_book_counts['dept_book_preference'] = dept_book_counts['count'] / dept_book_counts['total']
        merged = merged.merge(dept_book_counts[['DEPT', 'book_id', 'dept_book_preference']],
                              on=['DEPT', 'book_id'], how='left')
        merged['dept_book_preference'] = merged['dept_book_preference'].fillna(0)

        # 构建时间特征
        merged['借阅月份'] = merged['借阅时间'].dt.month
        merged['借阅星期'] = merged['借阅时间'].dt.weekday
        merged['借阅小时'] = merged['借阅时间'].dt.hour

        # 保存增强后的数据
        merged.to_csv(self.merged_borrows_path, index=False, encoding='utf-8')
        self.merged_data = merged

        self.logger.info(f"特征工程完成，耗时: {time.time() - start_time:.2f}秒")
        self.logger.info(f"增强后的数据: {len(merged)}条")

    def split_data(self, test_size=0.2, val_size=0.1):
        """划分训练集、验证集、测试集"""
        if self.merged_data is None:
            self.merged_data = pd.read_csv(self.merged_borrows_path, encoding='utf-8',
                                           parse_dates=['借阅时间', '还书时间'])

        # 按用户分组后分层抽样，保证每个用户的记录都分到不同集合
        user_groups = list(self.merged_data.groupby('user_id'))
        random.shuffle(user_groups)

        total_users = len(user_groups)
        test_users = int(total_users * test_size)
        val_users = int(total_users * val_size)

        test_data = pd.concat([group[1] for group in user_groups[:test_users]])
        val_data = pd.concat([group[1] for group in user_groups[test_users:test_users + val_users]])
        train_data = pd.concat([group[1] for group in user_groups[test_users + val_users:]])

        # 保存分割结果
        train_data.to_csv("data/train_data.csv", index=False, encoding='utf-8')
        val_data.to_csv("data/val_data.csv", index=False, encoding='utf-8')
        test_data.to_csv("data/test_data.csv", index=False, encoding='utf-8')

        self.logger.info(
            f"数据分割完成 - 训练集: {len(train_data)}条, 验证集: {len(val_data)}条, 测试集: {len(test_data)}条")
        return train_data, val_data, test_data


# 基础模型 - UserCF
class UserCF:
    def __init__(self, logger, top_k=20):
        self.logger = logger
        self.top_k = top_k
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_idx = {}
        self.item_idx = {}
        self.item_list = []

    def fit(self, data):
        """训练用户协同过滤模型"""
        self.logger.info(f"开始训练UserCF模型，近邻数量: {self.top_k}")
        start_time = time.time()

        # 构建用户-物品矩阵
        users = list(data['user_id'].unique())
        items = list(data['book_id'].unique())
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {i: j for j, i in enumerate(items)}
        self.item_list = items

        # 初始化用户-物品矩阵
        self.user_item_matrix = np.zeros((len(users), len(items)))
        for _, row in data.iterrows():
            u = self.user_idx[row['user_id']]
            i = self.item_idx[row['book_id']]
            self.user_item_matrix[u, i] = 1  # 二元交互

        # 计算用户相似度（余弦相似度）
        self.user_similarity = np.dot(self.user_item_matrix, self.user_item_matrix.T)
        norm = np.diag(self.user_similarity)
        self.user_similarity = self.user_similarity / np.sqrt(np.outer(norm, norm))

        self.logger.info(f"UserCF模型训练完成，耗时: {time.time() - start_time:.2f}秒")

    def predict(self, user_id, top_n=10):
        """为用户推荐物品"""
        if user_id not in self.user_idx:
            return []

        u = self.user_idx[user_id]
        # 找到最相似的top_k用户
        similar_users = np.argsort(self.user_similarity[u])[::-1][1:self.top_k + 1]

        # 计算推荐分数
        scores = np.zeros(len(self.item_list))
        for v in similar_users:
            scores += self.user_similarity[u, v] * self.user_item_matrix[v]

        # 排除用户已交互的物品
        user_items = np.where(self.user_item_matrix[u] > 0)[0]
        scores[user_items] = -1

        # 返回top_n推荐
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.item_list[i] for i in top_indices]


# LSTM序列推荐模型
class LSTMSeqRecModel(torch.nn.Module):
    def __init__(self, book_num, feature_dim, title_vector_dim, embedding_dim=128,
                 hidden_dim=256, num_layers=2, dropout=0.3):
        super(LSTMSeqRecModel, self).__init__()
        self.embedding = torch.nn.Embedding(book_num, embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim + feature_dim + title_vector_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, book_num)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, input_features):
        # input_ids: [batch_size, seq_len]
        # input_features: [batch_size, seq_len, feature_dim + title_vector_dim]

        emb = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        x = torch.cat([emb, input_features], dim=-1)  # 拼接嵌入和特征
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        x = self.dropout(x[:, -1, :])  # 取最后一个时间步
        output = self.fc(x)  # [batch_size, book_num]
        return output


class BorrowSequenceDataset(Dataset):
    def __init__(self, data, user_encoder, book_encoder, seq_len=5, title_vector_path=None):
        self.data = data.sort_values(['user_id', '借阅时间'])
        self.user_encoder = user_encoder
        self.book_encoder = book_encoder
        self.seq_len = seq_len
        self.title_vector_path = title_vector_path
        self.title_vec_dict = self._load_title_vectors()
        self.sequences = self._build_sequences()

    def _load_title_vectors(self):
        """加载书名向量"""
        if not self.title_vector_path or not os.path.exists(self.title_vector_path):
            self.title_vector_dim = 0
            return {}

        vec_df = pd.read_csv(self.title_vector_path)
        vector_cols = [col for col in vec_df.columns if col.startswith('title_vec_')]
        self.title_vector_dim = len(vector_cols)
        return {str(row['book_id']): row[vector_cols].values for _, row in vec_df.iterrows()}

    def _build_sequences(self):
        """构建用户借阅序列"""
        sequences = []
        for user_id, group in self.data.groupby('user_id'):
            books = group['book_id'].tolist()
            features = []

            # 提取特征
            for _, row in group.iterrows():
                # 基础特征
                base_feat = [row['交互权重'], row['dept_book_preference'], row['借阅时长'] / 30]

                # 书名向量特征
                book_id_str = str(row['book_id'])
                if book_id_str in self.title_vec_dict:
                    title_vec = self.title_vec_dict[book_id_str]
                    base_feat = np.concatenate([base_feat, title_vec])

                features.append(base_feat)

            # 生成序列样本
            for i in range(self.seq_len, len(books)):
                seq = books[i - self.seq_len:i]
                seq_feats = features[i - self.seq_len:i]
                target = books[i]
                sequences.append((seq, seq_feats, target))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, feats, target = self.sequences[idx]
        seq_encoded = self.book_encoder.transform(seq)
        target_encoded = self.book_encoder.transform([target])[0]
        return (torch.LongTensor(seq_encoded),
                torch.FloatTensor(feats),
                torch.LongTensor([target_encoded]))


# BERT序列推荐模型
class BertSeqRecModel(torch.nn.Module):
    def __init__(self, book_num, feature_dim, title_vec_dim, hidden_dim=256,
                 dropout=0.3, bert_model_name="bert-base-chinese"):
        super(BertSeqRecModel, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.feature_proj = torch.nn.Linear(feature_dim + title_vec_dim, 128)
        self.fusion = torch.nn.Linear(self.bert.config.hidden_size + 128, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, book_num)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, title_vecs, features):
        # input_ids: [batch_size, seq_len, max_bert_len]
        # attention_mask: [batch_size, seq_len, max_bert_len]
        # title_vecs: [batch_size, seq_len, title_vec_dim]
        # features: [batch_size, seq_len, feature_dim]

        batch_size, seq_len, max_bert_len = input_ids.shape

        # BERT编码
        input_ids = input_ids.view(-1, max_bert_len)  # [batch_size*seq_len, max_bert_len]
        attention_mask = attention_mask.view(-1, max_bert_len)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        bert_out = bert_out.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        bert_out = torch.mean(bert_out, dim=1)  # [batch_size, hidden_size]

        # 特征处理
        features = torch.cat([features, title_vecs], dim=-1)  # 拼接特征和书名向量
        feat_out = self.feature_proj(features)  # [batch_size, seq_len, 128]
        feat_out = torch.mean(feat_out, dim=1)  # [batch_size, 128]

        # 融合特征
        fusion_out = torch.cat([bert_out, feat_out], dim=-1)  # [batch_size, hidden_size+128]
        fusion_out = self.dropout(torch.relu(self.fusion(fusion_out)))

        # 预测
        output = self.fc(fusion_out)  # [batch_size, book_num]
        return output


class BertBorrowDataset(Dataset):
    def __init__(self, data, user_encoder, book_encoder, seq_len=5,
                 title_vector_path=None, bert_tokenizer=None, max_bert_len=10):
        self.data = data.sort_values(['user_id', '借阅时间'])
        self.user_encoder = user_encoder
        self.book_encoder = book_encoder
        self.seq_len = seq_len
        self.tokenizer = bert_tokenizer
        self.max_bert_len = max_bert_len
        self.title_vector_path = title_vector_path
        self.title_vec_dict = self._load_title_vectors()
        self.sequences = self._build_sequences()

    def _load_title_vectors(self):
        if not self.title_vector_path or not os.path.exists(self.title_vector_path):
            self.title_vec_dim = 0
            return {}

        vec_df = pd.read_csv(self.title_vector_path)
        vector_cols = [col for col in vec_df.columns if col.startswith('title_vec_')]
        self.title_vec_dim = len(vector_cols)
        return {str(row['book_id']): row[vector_cols].values for _, row in vec_df.iterrows()}

    def _build_sequences(self):
        sequences = []
        for user_id, group in self.data.groupby('user_id'):
            books = group['book_id'].tolist()
            titles = group['书名'].fillna('').tolist()
            features = []

            # 提取特征
            for _, row in group.iterrows():
                feat = [row['交互权重'], row['dept_book_preference'], row['借阅时长'] / 30]
                features.append(feat)

            # 生成序列样本
            for i in range(self.seq_len, len(books)):
                seq = books[i - self.seq_len:i]
                seq_titles = titles[i - self.seq_len:i]
                seq_feats = features[i - self.seq_len:i]
                target = books[i]
                sequences.append((seq, seq_titles, seq_feats, target))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, titles, feats, target = self.sequences[idx]

        # 编码目标
        target_encoded = self.book_encoder.transform([target])[0]

        # BERT输入
        bert_inputs = self.tokenizer(
            titles,
            padding='max_length',
            truncation=True,
            max_length=self.max_bert_len,
            return_tensors='pt'
        )
        input_ids = bert_inputs['input_ids']
        attention_mask = bert_inputs['attention_mask']

        # 书名向量
        title_vecs = []
        for book_id in seq:
            book_id_str = str(book_id)
            if book_id_str in self.title_vec_dict:
                title_vecs.append(self.title_vec_dict[book_id_str])
            else:
                title_vecs.append(np.zeros(self.title_vec_dim))

        return (torch.LongTensor(self.book_encoder.transform(seq)),
                input_ids,
                attention_mask,
                torch.FloatTensor(title_vecs),
                torch.FloatTensor(feats),
                torch.LongTensor([target_encoded]))


# 模型训练器
class ModelTrainer:
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.logger = self._init_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")

        # 数据处理器
        self.data_processor = DataProcessor(data_paths, self.logger)

        # 数据
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # 模型
        self.models = {}
        self.encoders = {
            'user': LabelEncoder(),
            'book': LabelEncoder()
        }

        # 结果保存路径
        self.model_save_dir = "model/saved_models"
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _init_logger(self):
        import logging
        logger = logging.getLogger("RecommendationTrainer")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("training.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def prepare_data(self, test_size=0.2, val_size=0.1, rebuild=False):
        """准备数据"""
        if not rebuild and os.path.exists("data/train_data.csv"):
            self.logger.info("加载已处理的数据...")
            self.train_data = pd.read_csv("data/train_data.csv", encoding='utf-8', parse_dates=['借阅时间', '还书时间'])
            self.val_data = pd.read_csv("data/val_data.csv", encoding='utf-8', parse_dates=['借阅时间', '还书时间'])
            self.test_data = pd.read_csv("data/test_data.csv", encoding='utf-8', parse_dates=['借阅时间', '还书时间'])
        else:
            self.data_processor.clean_data()
            self.data_processor.feature_engineering()
            self.train_data, self.val_data, self.test_data = self.data_processor.split_data(test_size, val_size)

        # 拟合编码器
        all_users = pd.concat(
            [self.train_data['user_id'], self.val_data['user_id'], self.test_data['user_id']]).unique()
        all_books = pd.concat(
            [self.train_data['book_id'], self.val_data['book_id'], self.test_data['book_id']]).unique()
        self.encoders['user'].fit(all_users)
        self.encoders['book'].fit(all_books)

        return self.train_data, self.val_data, self.test_data

    def train_user_cf(self, top_k=20):
        """训练UserCF模型"""
        model = UserCF(self.logger, top_k)
        model.fit(self.train_data)

        # 评估
        self.evaluate_model(model, "UserCF")

        # 保存模型
        model_path = os.path.join(self.model_save_dir, f"user_cf_topk{top_k}.pkl")
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.models['user_cf'] = model
        self.logger.info(f"UserCF模型保存至: {model_path}")
        return model

    def train_lstm(self, params):
        """训练LSTM模型"""
        self.logger.info("开始训练LSTM模型...")
        start_time = time.time()

        # 准备数据集
        train_dataset = BorrowSequenceDataset(
            self.train_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=params['seq_len'],
            title_vector_path=self.data_processor.title_vector_path
        )
        val_dataset = BorrowSequenceDataset(
            self.val_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=params['seq_len'],
            title_vector_path=self.data_processor.title_vector_path
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

        # 初始化模型
        book_num = len(self.encoders['book'].classes_)
        model = LSTMSeqRecModel(
            book_num=book_num,
            feature_dim=3,  # 基础特征维度
            title_vector_dim=train_dataset.title_vector_dim,
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)

        # 优化器和损失函数
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # 训练循环
        best_val_f1 = 0.0
        patience = params['patience']
        no_improve_epochs = 0

        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0
            train_preds = []
            train_trues = []

            for seq_ids, feats, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params['epochs']}"):
                seq_ids = seq_ids.to(self.device)
                feats = feats.to(self.device)
                target = target.squeeze().to(self.device)

                optimizer.zero_grad()
                outputs = model(seq_ids, feats)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_trues.extend(target.cpu().numpy())

            # 训练集评估
            train_precision = precision_score(train_trues, train_preds, average='macro')
            train_recall = recall_score(train_trues, train_preds, average='macro')
            train_f1 = f1_score(train_trues, train_preds, average='macro')

            # 验证集评估
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for seq_ids, feats, target in val_loader:
                    seq_ids = seq_ids.to(self.device)
                    feats = feats.to(self.device)
                    target = target.squeeze().to(self.device)

                    outputs = model(seq_ids, feats)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_trues.extend(target.cpu().numpy())

            val_precision = precision_score(val_trues, val_preds, average='macro')
            val_recall = recall_score(val_trues, val_preds, average='macro')
            val_f1 = f1_score(val_trues, val_preds, average='macro')

            self.logger.info(f"Epoch {epoch + 1} - 训练损失: {train_loss / len(train_loader):.4f}, "
                             f"训练F1: {train_f1:.4f}, 验证F1: {val_f1:.4f}")

            # 早停机制
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    self.logger.info(f"早停于第{epoch + 1}轮")
                    break

        # 加载最佳模型
        model.load_state_dict(best_model)

        # 在测试集评估
        self.evaluate_lstm(model, params['seq_len'])

        # 保存模型
        model_path = os.path.join(self.model_save_dir, f"lstm_seq{params['seq_len']}_f1{best_val_f1:.4f}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'user_encoder': self.encoders['user'],
            'book_encoder': self.encoders['book'],
            'params': params,
            'title_vector_dim': train_dataset.title_vector_dim
        }, model_path)
        self.models['lstm'] = model
        self.logger.info(f"LSTM模型保存至: {model_path}, 耗时: {time.time() - start_time:.2f}秒")
        return model

    def train_bert(self, params):
        """训练BERT模型"""
        self.logger.info("开始训练BERT模型...")
        start_time = time.time()

        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # 准备数据集
        train_dataset = BertBorrowDataset(
            self.train_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=params['seq_len'],
            title_vector_path=self.data_processor.title_vector_path,
            bert_tokenizer=tokenizer,
            max_bert_len=params['max_bert_len']
        )
        val_dataset = BertBorrowDataset(
            self.val_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=params['seq_len'],
            title_vector_path=self.data_processor.title_vector_path,
            bert_tokenizer=tokenizer,
            max_bert_len=params['max_bert_len']
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

        # 初始化模型
        book_num = len(self.encoders['book'].classes_)
        model = BertSeqRecModel(
            book_num=book_num,
            feature_dim=3,
            title_vec_dim=train_dataset.title_vec_dim,
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout']
        ).to(self.device)

        # 冻结BERT权重
        if params['freeze_bert']:
            for param in model.bert.parameters():
                param.requires_grad = False

        # 优化器和损失函数
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

        # 训练循环
        best_val_f1 = 0.0
        patience = params['patience']
        no_improve_epochs = 0

        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0
            train_preds = []
            train_trues = []

            for _, input_ids, attention_mask, title_vecs, features, target in tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{params['epochs']}"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                title_vecs = title_vecs.to(self.device)
                features = features.to(self.device)
                target = target.squeeze().to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, title_vecs, features)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_trues.extend(target.cpu().numpy())

            # 训练集评估
            train_precision = precision_score(train_trues, train_preds, average='macro')
            train_recall = recall_score(train_trues, train_preds, average='macro')
            train_f1 = f1_score(train_trues, train_preds, average='macro')

            # 验证集评估
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for _, input_ids, attention_mask, title_vecs, features, target in val_loader:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    title_vecs = title_vecs.to(self.device)
                    features = features.to(self.device)
                    target = target.squeeze().to(self.device)

                    outputs = model(input_ids, attention_mask, title_vecs, features)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_trues.extend(target.cpu().numpy())

            val_precision = precision_score(val_trues, val_preds, average='macro')
            val_recall = recall_score(val_trues, val_preds, average='macro')
            val_f1 = f1_score(val_trues, val_preds, average='macro')

            self.logger.info(f"Epoch {epoch + 1} - 训练损失: {train_loss / len(train_loader):.4f}, "
                             f"训练F1: {train_f1:.4f}, 验证F1: {val_f1:.4f}")

            # 早停机制
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    self.logger.info(f"早停于第{epoch + 1}轮")
                    break

        # 加载最佳模型
        model.load_state_dict(best_model)

        # 在测试集评估
        self.evaluate_bert(model, tokenizer, params['seq_len'], params['max_bert_len'])

        # 保存模型
        model_path = os.path.join(self.model_save_dir, f"bert_seq{params['seq_len']}_f1{best_val_f1:.4f}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'user_encoder': self.encoders['user'],
            'book_encoder': self.encoders['book'],
            'params': params,
            'title_vector_dim': train_dataset.title_vec_dim
        }, model_path)
        self.models['bert'] = model
        self.logger.info(f"BERT模型保存至: {model_path}, 耗时: {time.time() - start_time:.2f}秒")
        return model

    def train_xgb(self, params):
        """训练XGBoost模型"""
        self.logger.info("开始训练XGBoost模型...")
        start_time = time.time()

        # 构建正负样本
        train_samples = self._build_pos_neg_samples(self.train_data, params['neg_ratio'])
        val_samples = self._build_pos_neg_samples(self.val_data, params['neg_ratio'])

        # 特征和标签
        feature_cols = [col for col in train_samples.columns if col not in ['user_id', 'book_id', 'label']]
        X_train, y_train = train_samples[feature_cols], train_samples['label']
        X_val, y_val = val_samples[feature_cols], val_samples['label']

        # 训练模型
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            random_state=42
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        # 评估
        self.evaluate_rank_model(model, feature_cols, "XGBoost")

        # 保存模型
        model_path = os.path.join(self.model_save_dir,
                                  f"xgb_depth{params['max_depth']}_f1{self.models['xgb_metrics']['f1']:.4f}.pkl")
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_cols': feature_cols,
                'user_encoder': self.encoders['user'],
                'book_encoder': self.encoders['book'],
                'scaler': self.scaler
            }, f)
        self.models['xgb'] = model
        self.models['xgb_feature_cols'] = feature_cols
        self.logger.info(f"XGBoost模型保存至: {model_path}, 耗时: {time.time() - start_time:.2f}秒")
        return model

    def train_lightgbm(self, params):
        """训练LightGBM模型"""
        self.logger.info("开始训练LightGBM模型...")
        start_time = time.time()

        # 构建正负样本
        train_samples = self._build_pos_neg_samples(self.train_data, params['neg_ratio'])
        val_samples = self._build_pos_neg_samples(self.val_data, params['neg_ratio'])

        # 特征和标签
        feature_cols = [col for col in train_samples.columns if col not in ['user_id', 'book_id', 'label']]
        X_train, y_train = train_samples[feature_cols], train_samples['label']
        X_val, y_val = val_samples[feature_cols], val_samples['label']

        # 构建数据集
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # 训练模型
        model = lgb.train(
            {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': params['num_leaves'],
                'learning_rate': params['learning_rate'],
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
                'random_state': 42,
                'verbose': -1
            },
            lgb_train,
            num_boost_round=params['n_estimators'],
            valid_sets=[lgb_val],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # 评估
        self.evaluate_rank_model(model, feature_cols, "LightGBM")

        # 保存模型
        model_path = os.path.join(self.model_save_dir,
                                  f"lgb_leaves{params['num_leaves']}_f1{self.models['lgb_metrics']['f1']:.4f}.txt")
        model.save_model(model_path)
        # 保存相关信息
        import pickle
        with open(model_path.replace('.txt', '.pkl'), 'wb') as f:
            pickle.dump({
                'feature_cols': feature_cols,
                'user_encoder': self.encoders['user'],
                'book_encoder': self.encoders['book'],
                'scaler': self.scaler
            }, f)
        self.models['lightgbm'] = model
        self.models['lgb_feature_cols'] = feature_cols
        self.logger.info(f"LightGBM模型保存至: {model_path}, 耗时: {time.time() - start_time:.2f}秒")
        return model

    def _build_pos_neg_samples(self, data, neg_ratio=3):
        """构建正负样本（用于XGBoost和LightGBM）"""
        # 正样本
        pos_samples = data[['user_id', 'book_id']].drop_duplicates()
        pos_samples['label'] = 1

        # 负样本
        all_books = set(data['book_id'].unique())
        neg_samples = []
        for user_id, group in data.groupby('user_id'):
            user_books = set(group['book_id'])
            candidate_books = list(all_books - user_books)
            if not candidate_books:
                continue
            sample_size = min(len(user_books) * neg_ratio, len(candidate_books))
            neg_books = random.sample(candidate_books, sample_size)
            neg_samples.extend([(user_id, book, 0) for book in neg_books])

        neg_samples = pd.DataFrame(neg_samples, columns=['user_id', 'book_id', 'label'])
        samples = pd.concat([pos_samples, neg_samples], ignore_index=True)

        # 添加特征
        return self._add_features_to_samples(samples, data)

    def _add_features_to_samples(self, samples, data):
        """为样本添加特征"""
        # 用户特征
        user_feats = data.groupby('user_id').agg({
            'book_id': 'nunique',
            '借阅时长': ['mean', 'std'],
            '交互权重': 'sum'
        }).reset_index()
        user_feats.columns = ['user_id'] + [f'user_{c[0]}_{c[1]}' for c in user_feats.columns[1:]]

        # 物品特征
        item_feats = data.groupby('book_id').agg({
            'user_id': 'nunique',
            '借阅时长': ['mean', 'std'],
            '二级分类': 'first'
        }).reset_index()
        item_feats.columns = ['book_id'] + [f'item_{c[0]}_{c[1]}' for c in item_feats.columns[1:]]

        # 合并特征
        samples = samples.merge(user_feats, on='user_id', how='left')
        samples = samples.merge(item_feats, on='book_id', how='left')

        # 编码和归一化
        samples['user_id_encoded'] = self.encoders['user'].transform(samples['user_id'])
        samples['book_id_encoded'] = self.encoders['book'].transform(samples['book_id'])

        # 分类特征编码
        cat_cols = ['二级分类']
        for col in cat_cols:
            if col not in self.encoders['cat']:
                self.encoders['cat'][col] = LabelEncoder()
                self.encoders['cat'][col].fit(samples[col].fillna('未知'))
            samples[col] = samples[col].fillna('未知')
            samples[f'{col}_encoded'] = self.encoders['cat'][col].transform(samples[col])

        # 数值特征归一化
        num_cols = [col for col in samples.columns if
                    ('user_' in col or 'item_' in col) and
                    '_encoded' not in col]
        if not hasattr(self, 'scaler'):
            self.scaler = MinMaxScaler()
            self.scaler.fit(samples[num_cols].fillna(0))
        samples[num_cols] = self.scaler.transform(samples[num_cols].fillna(0))

        # 保留特征列
        keep_cols = ['user_id', 'book_id', 'label', 'user_id_encoded', 'book_id_encoded'] + \
                    [f'{col}_encoded' for col in cat_cols] + num_cols
        return samples[keep_cols]

    def evaluate_model(self, model, model_name, top_n=10):
        """评估推荐模型"""
        self.logger.info(f"评估{model_name}模型...")

        users = self.test_data['user_id'].unique()[:100]  # 取部分用户评估
        hits = 0
        total_pred = 0
        total_true = 0

        for user_id in tqdm(users, desc=f"评估{model_name}"):
            # 真实借阅
            true_books = set(self.test_data[self.test_data['user_id'] == user_id]['book_id'].unique())
            if not true_books:
                continue
            total_true += len(true_books)

            # 模型推荐
            pred_books = model.predict(user_id, top_n)
            total_pred += len(pred_books)

            # 命中数
            hits += len(set(pred_books) & true_books)

        precision = hits / total_pred if total_pred > 0 else 0
        recall = hits / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        self.logger.info(f"{model_name}评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.models[f'{model_name.lower()}_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return precision, recall, f1

    def evaluate_lstm(self, model, seq_len, top_n=10):
        """评估LSTM模型"""
        self.logger.info("评估LSTM模型...")

        # 构建测试数据集
        test_dataset = BorrowSequenceDataset(
            self.test_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=seq_len,
            title_vector_path=self.data_processor.title_vector_path
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for seq_ids, feats, target in test_loader:
                seq_ids = seq_ids.to(self.device)
                feats = feats.to(self.device)
                target = target.squeeze().to(self.device)

                outputs = model(seq_ids, feats)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(target.cpu().numpy())

        precision = precision_score(all_trues, all_preds, average='macro')
        recall = recall_score(all_trues, all_preds, average='macro')
        f1 = f1_score(all_trues, all_preds, average='macro')

        self.logger.info(f"LSTM评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.models['lstm_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return precision, recall, f1

    def evaluate_bert(self, model, tokenizer, seq_len, max_bert_len, top_n=10):
        """评估BERT模型"""
        self.logger.info("评估BERT模型...")

        # 构建测试数据集
        test_dataset = BertBorrowDataset(
            self.test_data,
            self.encoders['user'],
            self.encoders['book'],
            seq_len=seq_len,
            title_vector_path=self.data_processor.title_vector_path,
            bert_tokenizer=tokenizer,
            max_bert_len=max_bert_len
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for _, input_ids, attention_mask, title_vecs, features, target in test_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                title_vecs = title_vecs.to(self.device)
                features = features.to(self.device)
                target = target.squeeze().to(self.device)

                outputs = model(input_ids, attention_mask, title_vecs, features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(target.cpu().numpy())

        precision = precision_score(all_trues, all_preds, average='macro')
        recall = recall_score(all_trues, all_preds, average='macro')
        f1 = f1_score(all_trues, all_preds, average='macro')

        self.logger.info(f"BERT评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.models['bert_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return precision, recall, f1

    def evaluate_rank_model(self, model, feature_cols, model_name, top_n=10):
        """评估排序模型（XGBoost和LightGBM）"""
        self.logger.info(f"评估{model_name}模型...")

        users = self.test_data['user_id'].unique()[:100]  # 取部分用户评估
        hits = 0
        total_pred = 0
        total_true = 0

        for user_id in tqdm(users, desc=f"评估{model_name}"):
            # 真实借阅
            true_books = set(self.test_data[self.test_data['user_id'] == user_id]['book_id'].unique())
            if not true_books:
                continue
            total_true += len(true_books)

            # 候选集
            all_books = set(self.test_data['book_id'].unique())
            user_books = set(self.train_data[self.train_data['user_id'] == user_id]['book_id'])
            candidates = list(all_books - user_books)
            if not candidates:
                continue

            # 构建候选特征
            candidate_df = pd.DataFrame({'user_id': [user_id] * len(candidates), 'book_id': candidates})
            candidate_df = self._add_features_to_samples(candidate_df, self.train_data)

            # 预测
            if model_name == "XGBoost":
                scores = model.predict_proba(candidate_df[feature_cols])[:, 1]
            else:  # LightGBM
                scores = model.predict(candidate_df[feature_cols])

            # 排序推荐
            candidate_df['score'] = scores
            pred_books = candidate_df.sort_values('score', ascending=False)['book_id'].head(top_n).tolist()
            total_pred += len(pred_books)

            # 命中数
            hits += len(set(pred_books) & true_books)

        precision = hits / total_pred if total_pred > 0 else 0
        recall = hits / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        self.logger.info(f"{model_name}评估结果 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        self.models[f'{model_name.lower()}_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return precision, recall, f1

    def load_model(self, model_name, model_path):
        """加载预训练模型"""
        self.logger.info(f"加载{model_name}模型: {model_path}")

        if model_name == 'user_cf':
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models['user_cf'] = model

        elif model_name == 'lstm':
            checkpoint = torch.load(model_path, map_location=self.device)
            self.encoders['user'] = checkpoint['user_encoder']
            self.encoders['book'] = checkpoint['book_encoder']
            params = checkpoint['params']

            book_num = len(self.encoders['book'].classes_)
            model = LSTMSeqRecModel(
                book_num=book_num,
                feature_dim=3,
                title_vector_dim=checkpoint['title_vector_dim'],
                embedding_dim=params['embedding_dim'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models['lstm'] = model

        elif model_name == 'bert':
            checkpoint = torch.load(model_path, map_location=self.device)
            self.encoders['user'] = checkpoint['user_encoder']
            self.encoders['book'] = checkpoint['book_encoder']
            params = checkpoint['params']

            book_num = len(self.encoders['book'].classes_)
            model = BertSeqRecModel(
                book_num=book_num,
                feature_dim=3,
                title_vec_dim=checkpoint['title_vector_dim'],
                hidden_dim=params['hidden_dim'],
                dropout=params['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models['bert'] = model

        elif model_name == 'xgb':
            import pickle
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.models['xgb'] = data['model']
            self.models['xgb_feature_cols'] = data['feature_cols']
            self.encoders['user'] = data['user_encoder']
            self.encoders['book'] = data['book_encoder']
            self.scaler = data['scaler']

        elif model_name == 'lightgbm':
            model = lgb.Booster(model_file=model_path)
            import pickle
            with open(model_path.replace('.txt', '.pkl'), 'rb') as f:
                data = pickle.load(f)
            self.models['lightgbm'] = model
            self.models['lgb_feature_cols'] = data['feature_cols']
            self.encoders['user'] = data['user_encoder']
            self.encoders['book'] = data['book_encoder']
            self.scaler = data['scaler']

        else:
            raise ValueError(f"不支持的模型类型: {model_name}")

        return self.models[model_name]

    def recommend(self, model_name, user_id, top_n=10):
        """为用户推荐书籍"""
        if model_name not in self.models:
            raise ValueError(f"模型{model_name}未加载")

        model = self.models[model_name]

        if model_name == 'user_cf':
            return model.predict(user_id, top_n)

        elif model_name == 'lstm':
            # 构建用户序列
            user_data = self.train_data[self.train_data['user_id'] == user_id].sort_values('借阅时间')
            if len(user_data) < self.models['lstm_params']['seq_len']:
                return []

            books = user_data['book_id'].tolist()[-self.models['lstm_params']['seq_len']:]
            features = []

            # 提取特征
            for _, row in user_data.tail(self.models['lstm_params']['seq_len']).iterrows():
                base_feat = [row['交互权重'], row['dept_book_preference'], row['借阅时长'] / 30]

                # 书名向量
                title_vec_dict = BorrowSequenceDataset(
                    self.train_data, self.encoders['user'], self.encoders['book'],
                    title_vector_path=self.data_processor.title_vector_path
                ).title_vec_dict

                book_id_str = str(row['book_id'])
                if book_id_str in title_vec_dict:
                    base_feat = np.concatenate([base_feat, title_vec_dict[book_id_str]])

                features.append(base_feat)

            # 编码
            seq_encoded = self.encoders['book'].transform(books)

            # 预测
            model.eval()
            with torch.no_grad():
                input_ids = torch.LongTensor(seq_encoded).unsqueeze(0).to(self.device)
                input_feats = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = model(input_ids, input_feats)
                top_indices = torch.topk(output, top_n).indices.squeeze().cpu().numpy()

            # 解码
            if isinstance(top_indices, int):
                top_indices = [top_indices]
            return self.encoders['book'].inverse_transform(top_indices)

        elif model_name in ['xgb', 'lightgbm']:
            # 获取候选集
            all_books = set(self.train_data['book_id'].unique())
            user_books = set(self.train_data[self.train_data['user_id'] == user_id]['book_id'])
            candidates = list(all_books - user_books)
            if not candidates:
                return []

            # 构建特征
            candidate_df = pd.DataFrame({'user_id': [user_id] * len(candidates), 'book_id': candidates})
            candidate_df = self._add_features_to_samples(candidate_df, self.train_data)

            # 预测
            feature_cols = self.models['xgb_feature_cols'] if model_name == 'xgb' else self.models['lgb_feature_cols']
            if model_name == 'xgb':
                scores = model.predict_proba(candidate_df[feature_cols])[:, 1]
            else:
                scores = model.predict(candidate_df[feature_cols])

            # 排序
            candidate_df['score'] = scores
            return candidate_df.sort_values('score', ascending=False)['book_id'].head(top_n).tolist()

        elif model_name == 'bert':
            # 实现BERT推荐逻辑（类似LSTM）
            # 此处省略，可参考LSTM实现
            return []

        else:
            return []


# 主函数
def main():
    # 数据路径
    data_paths = {
        "books": "./data/item.csv",
        "borrows": "./data/inter_preliminary.csv",
        "users": "./data/user.csv"
    }

    # 初始化训练器
    trainer = ModelTrainer(data_paths)

    # 准备数据
    rebuild_data = input("是否重新处理数据？(y/n): ").lower() == 'y'
    trainer.prepare_data(rebuild=rebuild_data)

    # 选择操作
    while True:
        print("\n===== 推荐模型训练与评估系统 =====")
        print("1. 训练UserCF模型")
        print("2. 训练LSTM模型")
        print("3. 训练BERT模型")
        print("4. 训练XGBoost模型")
        print("5. 训练LightGBM模型")
        print("6. 加载模型")
        print("7. 推荐书籍")
        print("8. 退出")

        choice = input("请选择操作: ")

        if choice == '1':
            top_k = int(input("请输入近邻数量(默认20): ") or 20)
            trainer.train_user_cf(top_k)

        elif choice == '2':
            params = {
                'seq_len': int(input("请输入序列长度(默认5): ") or 5),
                'embedding_dim': int(input("请输入嵌入维度(默认128): ") or 128),
                'hidden_dim': int(input("请输入隐藏层维度(默认256): ") or 256),
                'num_layers': int(input("请输入LSTM层数(默认2): ") or 2),
                'dropout': float(input("请输入dropout比例(默认0.3): ") or 0.3),
                'batch_size': int(input("请输入批次大小(默认32): ") or 32),
                'lr': float(input("请输入学习率(默认0.001): ") or 0.001),
                'epochs': int(input("请输入训练轮数(默认30): ") or 30),
                'patience': int(input("请输入早停耐心值(默认3): ") or 3)
            }
            trainer.train_lstm(params)

        elif choice == '3':
            params = {
                'seq_len': int(input("请输入序列长度(默认5): ") or 5),
                'max_bert_len': int(input("请输入书名最大长度(默认10): ") or 10),
                'hidden_dim': int(input("请输入隐藏层维度(默认256): ") or 256),
                'dropout': float(input("请输入dropout比例(默认0.3): ") or 0.3),
                'batch_size': int(input("请输入批次大小(默认16): ") or 16),
                'lr': float(input("请输入学习率(默认2e-5): ") or 2e-5),
                'epochs': int(input("请输入训练轮数(默认10): ") or 10),
                'patience': int(input("请输入早停耐心值(默认3): ") or 3),
                'freeze_bert': input("是否冻结BERT权重(y/n): ").lower() == 'y'
            }
            trainer.train_bert(params)

        elif choice == '4':
            params = {
                'max_depth': int(input("请输入最大深度(默认6): ") or 6),
                'learning_rate': float(input("请输入学习率(默认0.1): ") or 0.1),
                'n_estimators': int(input("请输入树的数量(默认200): ") or 200),
                'neg_ratio': int(input("请输入负样本比例(默认3): ") or 3)
            }
            trainer.train_xgb(params)

        elif choice == '5':
            params = {
                'num_leaves': int(input("请输入叶子节点数(默认31): ") or 31),
                'learning_rate': float(input("请输入学习率(默认0.05): ") or 0.05),
                'n_estimators': int(input("请输入树的数量(默认200): ") or 200),
                'neg_ratio': int(input("请输入负样本比例(默认3): ") or 3)
            }
            trainer.train_lightgbm(params)

        elif choice == '6':
            model_type = input("请输入模型类型(user_cf/lstm/bert/xgb/lightgbm): ")
            model_path = input("请输入模型路径: ")
            if os.path.exists(model_path):
                trainer.load_model(model_type, model_path)
                print("模型加载成功")
            else:
                print("模型路径不存在")

        elif choice == '7':
            model_type = input("请输入模型类型(user_cf/lstm/bert/xgb/lightgbm): ")
            user_id = input("请输入用户ID: ")
            try:
                user_id = int(user_id)
            except:
                pass
            top_n = int(input("请输入推荐数量(默认10): ") or 10)

            try:
                recommendations = trainer.recommend(model_type, user_id, top_n)
                print(f"为用户{user_id}推荐的书籍ID: {recommendations}")
            except Exception as e:
                print(f"推荐失败: {str(e)}")

        elif choice == '8':
            print("退出系统")
            break

        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()