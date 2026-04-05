import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np


class BertSeqRecModel(nn.Module):
    def __init__(self,
                 book_num,  # 图书总数
                 feature_dim=3,  # 基础特征维度（如交互权重等）
                 title_vec_dim=128,  # 词向量维度（与你的词向量模型匹配）
                 bert_model_name="bert-base-chinese",  # 中文BERT预训练模型
                 hidden_dim=256,  # 融合层隐藏维度
                 dropout=0.3,
                 freeze_bert=False  # 是否冻结BERT权重（小数据建议冻结）
                 ):
        super().__init__()
        # 1. BERT文本编码器（处理书名等文本）
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size  # BERT输出维度（默认768）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False  # 冻结BERT参数

        # 2. 特征融合层（融合BERT编码、词向量、基础特征）
        self.fusion_dim = self.bert_hidden_dim + title_vec_dim + feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. 序列建模层（用Transformer处理用户借阅序列）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,  # 多头注意力头数
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True  # 批次维度在前
            ),
            num_layers=2  # Transformer层数
        )

        # 4. 输出层（预测下一本图书）
        self.output_layer = nn.Linear(hidden_dim, book_num)

    def forward(self,
                input_ids,  # BERT输入ID: [batch_size, seq_len, bert_input_len]
                attention_mask,  # BERT注意力掩码: [batch_size, seq_len, bert_input_len]
                title_vecs,  # 词向量特征: [batch_size, seq_len, title_vec_dim]
                features  # 基础特征: [batch_size, seq_len, feature_dim]
                ):
        batch_size, seq_len, bert_input_len = input_ids.shape

        # 1. BERT编码每本书的文本（展平序列维度方便批量处理）
        input_ids_flat = input_ids.view(-1, bert_input_len)  # [batch*seq_len, bert_input_len]
        attention_mask_flat = attention_mask.view(-1, bert_input_len)  # [batch*seq_len, bert_input_len]
        bert_outputs = self.bert(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        bert_embeds = bert_outputs.last_hidden_state[:, 0, :]  # 取[CLS] token的输出: [batch*seq_len, 768]
        bert_embeds = bert_embeds.view(batch_size, seq_len, self.bert_hidden_dim)  # 恢复序列维度: [batch, seq_len, 768]

        # 2. 融合所有特征（BERT编码 + 词向量 + 基础特征）
        fused_features = torch.cat([bert_embeds, title_vecs, features], dim=-1)  # [batch, seq_len, fusion_dim]
        fused_features = self.fusion_layer(fused_features)  # [batch, seq_len, hidden_dim]

        # 3. Transformer处理序列
        seq_output = self.transformer(fused_features)  # [batch, seq_len, hidden_dim]

        # 4. 用最后一个位置的输出预测下一本书
        final_output = seq_output[:, -1, :]  # [batch, hidden_dim]
        logits = self.output_layer(final_output)  # [batch, book_num]
        return logits


class BertBorrowDataset(torch.utils.data.Dataset):
    """适配BERT模型的数据集（融合词向量和文本特征）"""

    def __init__(self,
                 inter_df,  # 预处理后的交互数据（含user_id, book_id, 基础特征, 书名文本等）
                 user_encoder,  # 用户编码器
                 book_encoder,  # 图书编码器
                 seq_len=5,  # 序列长度
                 title_vector_path="data/item_with_title_vector.csv",  # 词向量文件路径
                 bert_tokenizer=None,  # BERT分词器
                 max_bert_len=10  # 单本书名的最大分词长度
                 ):
        self.inter_df = inter_df
        self.user_encoder = user_encoder
        self.book_encoder = book_encoder
        self.seq_len = seq_len
        self.max_bert_len = max_bert_len
        self.tokenizer = bert_tokenizer or BertTokenizer.from_pretrained("bert-base-chinese")

        # 加载词向量（与你的词向量模型对应）
        self.title_vec_dict = self._load_title_vectors(title_vector_path)
        self.title_vec_dim = len(next(iter(self.title_vec_dict.values()))) if self.title_vec_dict else 0

        # 按用户分组并排序序列
        self.user_seqs = self._build_user_sequences()

    def _load_title_vectors(self, path):
        """加载词向量（格式：book_id -> 向量）"""
        import pandas as pd
        vec_df = pd.read_csv(path)
        vector_cols = [col for col in vec_df.columns if col.startswith('title_vec_')]
        return {
            str(row['book_id']): row[vector_cols].values.astype(np.float32)
            for _, row in vec_df.iterrows()
        }

    def _build_user_sequences(self):
        """按用户构建借阅序列（含图书ID、基础特征、书名文本）"""
        user_seqs = {}
        # 按用户分组，按借阅时间排序
        grouped = self.inter_df.groupby('user_id').apply(
            lambda x: x.sort_values('借阅时间').reset_index(drop=True)
        )
        for user_id, group in grouped.groupby(level=0):
            # 提取序列所需字段（根据你的特征名调整）
            seq = {
                'book_ids': group['book_id'].tolist(),
                'features': group[['交互权重', 'dept_book_preference', 'dept_cat1_preference']].values,  # 基础特征
                'titles': group['题名'].fillna('').tolist()  # 书名文本
            }
            user_seqs[user_id] = seq
        return user_seqs

    def __len__(self):
        """总样本数 = 所有用户的有效序列数之和"""
        return sum(len(seq['book_ids']) - self.seq_len for seq in self.user_seqs.values()
                   if len(seq['book_ids']) > self.seq_len)

    def __getitem__(self, idx):
        """获取一个样本：(输入序列, 目标图书)"""
        # 遍历找到对应的用户序列（效率可优化，此处简化）
        for user_id, seq in self.user_seqs.items():
            book_ids = seq['book_ids']
            if len(book_ids) <= self.seq_len:
                continue
            # 取滑动窗口作为样本
            if idx < len(book_ids) - self.seq_len:
                # 输入序列（前seq_len本书）
                input_books = book_ids[idx:idx + self.seq_len]
                input_features = seq['features'][idx:idx + self.seq_len]
                input_titles = seq['titles'][idx:idx + self.seq_len]
                # 目标图书（第seq_len+1本）
                target_book = book_ids[idx + self.seq_len]

                # 1. 编码图书ID
                input_ids_encoded = self.book_encoder.transform(input_books)
                target_encoded = self.book_encoder.transform([target_book])[0]

                # 2. 处理BERT输入（书名分词）
                bert_inputs = self.tokenizer(
                    input_titles,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_bert_len,
                    return_tensors='pt'
                )
                input_ids = bert_inputs['input_ids']  # [seq_len, max_bert_len]
                attention_mask = bert_inputs['attention_mask']  # [seq_len, max_bert_len]

                # 3. 获取词向量
                title_vecs = np.array([
                    self.title_vec_dict.get(str(book_id), np.zeros(self.title_vec_dim))
                    for book_id in input_books
                ], dtype=np.float32)

                # 4. 基础特征（已在__init__中提取）
                features = input_features.astype(np.float32)

                return (
                    torch.LongTensor(input_ids_encoded),  # 图书ID编码（备用，可选）
                    input_ids,  # BERT输入ID
                    attention_mask,  # BERT注意力掩码
                    torch.FloatTensor(title_vecs),  # 词向量
                    torch.FloatTensor(features),  # 基础特征
                    torch.LongTensor([target_encoded])  # 目标图书
                )
            else:
                idx -= len(book_ids) - self.seq_len
        raise IndexError("样本索引超出范围")