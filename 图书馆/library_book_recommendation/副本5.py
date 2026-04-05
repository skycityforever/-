import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BorrowSequenceDataset(Dataset):
    """构建借阅序列数据集：将用户的历史借阅记录按时间排序，生成输入序列与目标图书"""

    def __init__(self, inter_df, user_encoder, book_encoder, seq_len=5):
        self.seq_len = seq_len  # 输入序列长度（取用户最近5次借阅记录）
        self.user_encoder = user_encoder  # 用户ID编码（str→int）
        self.book_encoder = book_encoder  # 图书ID编码（str→int）

        # 1. 按用户分组，按借阅时间排序（确保时序性）
        self.user_groups = inter_df.sort_values('借阅时间').groupby('user_id')
        # 2. 生成每个用户的借阅图书序列
        self.user_seq_list = []
        for user_id, group in self.user_groups:
            # 保留图书ID和对应的特征（每一行是一本图书+其特征）
            book_with_features = group[['book_id', '交互权重', 'dept_book_preference', 'dept_cat1_preference']].to_dict(
                'records')
            if len(book_with_features) >= self.seq_len + 1:
                self.user_seq_list.append((user_id, book_with_features))

    def __len__(self):
        return sum(len(seq) - self.seq_len for _, seq in self.user_seq_list)

    def __getitem__(self, idx):
        # 生成单个训练样本
        current_idx = 0
        for user_id, seq in self.user_seq_list:
            seq_len_total = len(seq) - self.seq_len
            if current_idx + seq_len_total > idx:
                """
                __getitem__ 的遍历逻辑是「按用户逐个查找」：先遍历第一个用户的所有样本，再遍历第二个，以此类推。当找到 idx 所属的用户后，
                idx - current_idx 就是「这个样本在当前用户序列内的局部起始位置」，用来确定从哪里截取 seq_len 长度的输入序列。
                """
                seq_start = idx - current_idx
                # 提取输入序列（含特征）和目标图书ID
                input_seq = seq[seq_start:seq_start + self.seq_len]
                target_book = seq[seq_start + self.seq_len]['book_id']

                # 编码图书ID，并提取特征
                input_book_ids = [x['book_id'] for x in input_seq]
                input_encoded = torch.tensor([self.book_encoder.transform([b])[0] for b in input_book_ids],
                                             dtype=torch.long)
                # 提取特征（交互权重、院系偏好等，转为张量）
                input_features = torch.tensor(
                    [[x['交互权重'], x['dept_book_preference'], x['dept_cat1_preference']] for x in input_seq],
                    dtype=torch.float32
                )
                target_encoded = torch.tensor(self.book_encoder.transform([target_book])[0], dtype=torch.long)

                # 返回：图书ID编码 + 特征
                return input_encoded, input_features, target_encoded
            current_idx += seq_len_total
        raise IndexError("索引超出范围")


class LSTMSeqRecModel(nn.Module):
    def __init__(self, book_num, feature_dim=3, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.book_embedding = nn.Embedding(book_num, embedding_dim)  # 图书嵌入
        self.feature_dim = feature_dim  # 新增特征的维度（这里是3个特征）
        self.lstm = nn.LSTM(
            input_size=embedding_dim + feature_dim,  # 输入=图书嵌入+特征
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, book_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, input_features):
        # 1. 图书嵌入：(batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        embedded = self.book_embedding(input_ids)
        # 2. 拼接特征：(batch_size, seq_len, embedding_dim + feature_dim)
        lstm_input = torch.cat([embedded, input_features], dim=2)
        # 3. LSTM编码：取最后一步输出
        lstm_out, _ = self.lstm(lstm_input)
        seq_repr = lstm_out[:, -1, :]
        # 4. 预测
        logits = self.fc(seq_repr)
        probs = self.softmax(logits)
        return probs
