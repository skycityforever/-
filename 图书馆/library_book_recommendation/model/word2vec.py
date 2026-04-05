import os
import re
import numpy as np
import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm


class BookTitleWord2Vec:
    """基于Word2Vec的书名特征向量生成器（适配train.csv和item.csv）"""

    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        """
        初始化Word2Vec参数

        Args:
            vector_size: 词向量维度
            window: 上下文窗口大小
            min_count: 最小词频（低于此值的词不参与训练）
            workers: 训练时的线程数
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None  # Word2Vec模型实例
        self.vocab = None  # 词汇表（set类型）

    def _preprocess_title(self, title):
        """
        预处理书名：清洗文本、分词（移除停用词过滤）

        Args:
            title: 原始书名（字符串）

        Returns:
            处理后的词语列表（如：["深度学习", "入门"]）
        """
        if not isinstance(title, str) or pd.isna(title):
            return []

        # 1. 清洗文本：保留中文字符、字母、数字，去除特殊符号
        title_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', title)
        # 2. 分词（中文用jieba，英文用simple_preprocess）
        words = jieba.lcut(title_clean) if any('\u4e00' <= c <= '\u9fa5' for c in title_clean) else simple_preprocess(
            title_clean)
        # 3. 仅过滤长度为1的无意义词（保留所有有效词汇）
        words_filtered = [word for word in words if len(word) > 1]
        return words_filtered

    def load_titles_from_csv(self, item_csv_path, title_col="题名"):
        """
        从item.csv中读取所有书名（图书信息表通常包含书名）

        Args:
            item_csv_path: item.csv文件路径
            title_col: 书名所在列名（根据你的实际列名修改）

        Returns:
            去重后的书名列表
        """
        if not os.path.exists(item_csv_path):
            raise FileNotFoundError(f"文件不存在：{item_csv_path}")

        # 读取item.csv，仅保留书名列
        df = pd.read_csv(item_csv_path, usecols=[title_col])
        # 去重并过滤空值
        titles = df[title_col].dropna().unique().tolist()
        print(f"从{item_csv_path}中读取到 {len(titles)} 个不重复的书名")
        return titles

    def train(self, titles, model_save_path=None):
        """
        训练Word2Vec模型

        Args:
            titles: 书名列表（从item.csv读取）
            model_save_path: 模型保存路径（如"model/saved_models/title_word2vec.model"）
        """
        # 1. 预处理所有书名，生成训练语料
        print("预处理书名数据...")
        corpus = [self._preprocess_title(title) for title in tqdm(titles, desc="预处理进度")]
        # 过滤空列表（避免影响训练）
        corpus = [words for words in corpus if len(words) > 0]

        if not corpus:
            raise ValueError("预处理后无有效训练数据，请检查书名格式或列名是否正确")

        # 2. 训练Word2Vec模型
        print(f"开始训练Word2Vec模型（向量维度：{self.vector_size}）...")
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10  # 训练轮次，可调整
        )
        self.vocab = set(self.model.wv.index_to_key)  # 更新词汇表
        print(f"训练完成，词汇表大小：{len(self.vocab)}")

        # 3. 保存模型（如果指定路径）
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            self.model.save(model_save_path)
            print(f"模型已保存至：{model_save_path}")

    def load_model(self, model_path):
        """加载预训练的Word2Vec模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        self.model = Word2Vec.load(model_path)
        self.vocab = set(self.model.wv.index_to_key)
        self.vector_size = self.model.vector_size  # 同步向量维度
        print(f"模型加载成功，词汇表大小：{len(self.vocab)}，向量维度：{self.vector_size}")

    def get_title_vector(self, title):
        """
        将单本书名转换为特征向量（取所有词向量的平均值）

        Args:
            title: 原始书名（字符串）

        Returns:
            书名向量（numpy数组，形状为[vector_size,]）
        """
        if self.model is None:
            raise RuntimeError("请先训练或加载模型")

        words = self._preprocess_title(title)
        if not words:
            # 若书名预处理后无有效词，返回零向量
            return np.zeros(self.vector_size, dtype=np.float32)

        # 收集词向量（忽略不在词汇表中的词）
        word_vectors = []
        for word in words:
            if word in self.vocab:
                word_vectors.append(self.model.wv[word])

        if not word_vectors:
            # 若所有词都不在词汇表中，返回零向量
            return np.zeros(self.vector_size, dtype=np.float32)

        # 取平均作为书名向量（可根据需求改为求和、加权平均等）
        return np.mean(word_vectors, axis=0).astype(np.float32)

    def batch_get_title_vectors(self, titles):
        """
        批量将书名转换为特征向量

        Args:
            titles: 书名列表

        Returns:
            向量矩阵（numpy数组，形状为[len(titles), vector_size]）
        """
        return np.array([self.get_title_vector(title) for title in tqdm(titles, desc="生成书名向量")], dtype=np.float32)

    def generate_title_vectors_for_csv(self, input_csv_path, output_csv_path, title_col="书名", id_col="book_id"):
        """
        为CSV文件中的图书生成书名向量并保存（适配item.csv或train.csv）

        Args:
            input_csv_path: 输入CSV路径（如item.csv）
            output_csv_path: 输出CSV路径（含书名向量）
            title_col: 书名列名
            id_col: 图书ID列名（用于关联向量和图书）
        """
        if self.model is None:
            raise RuntimeError("请先训练或加载模型")

        # 读取输入CSV
        df = pd.read_csv(input_csv_path)
        if title_col not in df.columns or id_col not in df.columns:
            raise KeyError(f"输入CSV中缺少列：{title_col} 或 {id_col}")

        # 批量生成书名向量
        print(f"为{len(df)}条记录生成书名向量...")
        title_vectors = self.batch_get_title_vectors(df[title_col].tolist())

        # 将向量拆分为多列（便于后续模型输入）
        vector_cols = [f"title_vec_{i}" for i in range(self.vector_size)]
        vector_df = pd.DataFrame(title_vectors, columns=vector_cols)

        # 合并图书ID和向量，保存到输出CSV
        result_df = pd.concat([df[[id_col, title_col]], vector_df], axis=1)
        result_df.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"含书名向量的数据已保存至：{output_csv_path}")


# 示例用法（适配你的数据集）
if __name__ == "__main__":
    # 配置文件路径（根据你的实际路径修改）
    ITEM_CSV_PATH = "../data/item.csv"  # 图书信息表（含书名）
    MODEL_SAVE_PATH = "./model/saved_models/title_word2vec.model"  # 模型保存路径
    OUTPUT_VECTOR_CSV = "../data/item_with_title_vector.csv"  # 含向量的图书数据

    # 1. 初始化Word2Vec实例
    w2v = BookTitleWord2Vec(
        vector_size=128,  # 可调整向量维度
        window=3,  # 上下文窗口大小
        min_count=1,  # 最小词频（数据集小时设为1）
        workers=4
    )

    # 2. 从item.csv读取书名并训练模型
    titles = w2v.load_titles_from_csv(ITEM_CSV_PATH, title_col="题名")  # 确保title_col与你的CSV列名一致
    w2v.train(titles, model_save_path=MODEL_SAVE_PATH)

    # 3. （可选）加载预训练模型（后续使用时跳过训练步骤）
    # w2v = BookTitleWord2Vec()
    # w2v.load_model(MODEL_SAVE_PATH)

    # 4. 为item.csv生成书名向量并保存（供后续模型使用）
    w2v.generate_title_vectors_for_csv(
        input_csv_path=ITEM_CSV_PATH,
        output_csv_path=OUTPUT_VECTOR_CSV,
        title_col="题名",
        id_col="book_id"  # 确保id_col与你的CSV中图书ID列名一致
    )

    # 5. （可选）为train.csv中的书名生成向量（如果train.csv含书名）
    # w2v.generate_title_vectors_for_csv(
    #     input_csv_path=TRAIN_CSV_PATH,
    #     output_csv_path="./data/train_with_title_vector.csv",
    #     title_col="书名",
    #     id_col="user_id"  # 或train.csv中的其他主键列
    # )