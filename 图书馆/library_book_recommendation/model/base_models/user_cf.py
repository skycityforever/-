import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserCF:
    def __init__(self, top_k=20):
        self.top_k = top_k  # 选取Top-K相似用户
        self.user_similarity = None  # 用户相似度矩阵
        self.user_item_matrix = None  # 用户-物品交互矩阵
        # 新增：保存用户/图书索引映射（关键，解决属性不存在问题）
        self.user_idx = None  # {user_id: 索引}
        self.book_idx = None  # {book_id: 索引}
        self.book_list = None  # 图书ID列表（用于索引转原始ID）

    def fit(self, train_inter):
        """整合训练流程：构建矩阵→计算相似度（对外暴露的训练接口）"""
        # 1. 获取唯一用户和图书列表
        user_list = train_inter['user_id'].unique()
        self.book_list = train_inter['book_id'].unique()  # 保存图书列表（用于后续映射）

        # 2. 构建用户-物品矩阵，并保存索引映射
        self.user_idx, self.book_idx = self.build_user_item_matrix(train_inter, user_list, self.book_list)

        # 3. 计算用户相似度
        self.compute_user_similarity()

    def build_user_item_matrix(self, train_inter, user_list, book_list):
        """构建用户-物品交互矩阵（加权），并返回索引映射"""
        # 构建索引映射（同时保存到实例属性）
        user_idx = {user: i for i, user in enumerate(user_list)}
        book_idx = {book: j for j, book in enumerate(book_list)}

        # 初始化交互矩阵（用户数×图书数）
        self.user_item_matrix = np.zeros((len(user_list), len(book_list)))

        # 填充交互权重（用特征工程后的“交互权重”作为值）
        for _, row in train_inter.iterrows():
            u_idx = user_idx[row['user_id']]
            b_idx = book_idx[row['book_id']]
            self.user_item_matrix[u_idx, b_idx] = row['交互权重']  # 复用特征工程的权重

        return user_idx, book_idx

    def compute_user_similarity(self):
        """计算用户余弦相似度（基于交互矩阵）"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        return self.user_similarity

    def predict(self, user_id):
        """预测用户对图书的兴趣得分，返回Top-1推荐图书（简化参数，直接用实例属性）"""
        # 1. 检查用户是否在训练集中（新用户返回None，可后续补充冷启动逻辑）
        if user_id not in self.user_idx:
            return None

        # 2. 获取用户索引
        u_idx = self.user_idx[user_id]

        # 3. 找到Top-K相似用户（排除自己）
        similar_users = np.argsort(self.user_similarity[u_idx])[::-1][1:self.top_k + 1]

        # 4. 计算图书兴趣得分（相似用户的交互权重×相似度加权）
        book_scores = np.zeros(len(self.book_list))
        for s_user in similar_users:
            # 相似度×相似用户对图书的交互权重
            book_scores += self.user_similarity[u_idx, s_user] * self.user_item_matrix[s_user]

        # 5. 过滤用户已借阅过的图书（不推荐已借过的）
        user_borrowed = np.where(self.user_item_matrix[u_idx] > 0)[0]  # 已借图书的索引
        book_scores[user_borrowed] = -np.inf  # 置为负无穷，不影响最大值选择

        # 6. 返回得分最高的图书（原始ID）
        top_book_idx = np.argmax(book_scores)
        return self.book_list[top_book_idx]