import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class BorrowSequenceDataset(Dataset):
    """构建借阅序列数据集：整合书名向量特征"""

    def __init__(self, inter_df, user_encoder, book_encoder, seq_len=5, title_vector_path=None):
        self.seq_len = seq_len  # 输入序列长度
        self.user_encoder = user_encoder  # 用户ID编码器
        self.book_encoder = book_encoder  # 图书ID编码器

        # 新增：加载书名向量（key: book_id, value: 向量数组）
        self.book_title_vector = self._load_title_vectors(title_vector_path) if title_vector_path else None
        self.title_vector_dim = len(next(iter(self.book_title_vector.values()))) if self.book_title_vector else 0

        # 按用户分组并按时间排序
        self.user_groups = inter_df.sort_values('借阅时间').groupby('user_id')
        self.user_seq_list = []
        for user_id, group in self.user_groups:
            # 保留图书ID和原有特征
            book_with_features = group[['book_id', '交互权重', 'dept_book_preference', 'dept_cat1_preference']].to_dict(
                'records')
            # 过滤序列长度不足的用户（需满足：序列长度 >= seq_len + 1）
            if len(book_with_features) >= self.seq_len + 1:
                self.user_seq_list.append((user_id, book_with_features))

    def _load_title_vectors(self, csv_path):
        """加载书名向量CSV，返回 {book_id: 向量} 的字典"""
        if not csv_path or not pd.io.common.file_exists(csv_path):
            raise FileNotFoundError(f"书名向量文件不存在：{csv_path}")

        df = pd.read_csv(csv_path)
        # 假设图书ID列是'book_id'，向量列是'title_vec_0'到'title_vec_{n-1}'
        book_ids = df['book_id'].tolist()
        vector_cols = [col for col in df.columns if col.startswith('title_vec_')]
        vectors = df[vector_cols].values  # 形状：(num_books, vector_dim)

        # 构建映射（确保book_id是字符串类型，与交互数据中的一致）
        return {str(book_id): vec for book_id, vec in zip(book_ids, vectors)}

    def __len__(self):
        return sum(len(seq) - self.seq_len for _, seq in self.user_seq_list)

    def __getitem__(self, idx):
        current_idx = 0
        for user_id, seq in self.user_seq_list:
            seq_len_total = len(seq) - self.seq_len
            if current_idx + seq_len_total > idx:
                seq_start = idx - current_idx
                input_seq = seq[seq_start:seq_start + self.seq_len]
                target_book = seq[seq_start + self.seq_len]['book_id']

                # 1. 编码图书ID
                input_book_ids = [x['book_id'] for x in input_seq]
                input_encoded = torch.tensor(
                    [self.book_encoder.transform([str(b)])[0] for b in input_book_ids],
                    dtype=torch.long
                )

                # 2. 提取原有特征（交互权重、院系偏好）
                base_features = [[x['交互权重'], x['dept_book_preference'], x['dept_cat1_preference']] for x in
                                 input_seq]

                # 3. 拼接书名向量特征
                if self.book_title_vector:
                    for i, x in enumerate(input_seq):
                        book_id_str = str(x['book_id'])
                        # 若图书无对应向量，用零向量填充
                        title_vec = self.book_title_vector.get(book_id_str, np.zeros(self.title_vector_dim))
                        base_features[i].extend(title_vec)  # 拼接：[原有3维 + 书名向量N维]

                # 转为张量
                input_features = torch.tensor(base_features, dtype=torch.float32)

                # 4. 编码目标图书ID
                target_encoded = torch.tensor(
                    self.book_encoder.transform([str(target_book)]),
                    dtype=torch.long
                ).squeeze()

                return input_encoded, input_features, target_encoded
            current_idx += seq_len_total
        raise IndexError("索引超出范围")


class LSTMSeqRecModel(nn.Module):
    def __init__(self, book_num, feature_dim=3, title_vector_dim=0, embedding_dim=128, hidden_dim=256, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.book_embedding = nn.Embedding(book_num, embedding_dim)  # 图书嵌入层
        self.total_feature_dim = feature_dim + title_vector_dim  # 总特征维度（原有3维 + 书名向量维度）

        # LSTM输入维度 = 图书嵌入维度 + 总特征维度
        self.lstm = nn.LSTM(
            input_size=embedding_dim + self.total_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, book_num)  # 输出层：预测图书ID
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, input_features):
        # 1. 图书嵌入：(batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        embedded = self.book_embedding(input_ids)

        # 2. 拼接嵌入和特征：(batch_size, seq_len, embedding_dim + total_feature_dim)
        lstm_input = torch.cat([embedded, input_features], dim=2)

        # 3. LSTM编码：取最后一个时间步的输出作为序列表征
        lstm_out, _ = self.lstm(lstm_input)
        seq_repr = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # 4. 预测图书概率分布
        logits = self.fc(seq_repr)
        probs = self.softmax(logits)
        return probs