import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import re
import jieba
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import random
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 继承自你的日志工具
from 图书馆.library_book_recommendation.utils.logger import Logger


class XGBoostRecommender:
    def __init__(self, data_paths, device='cpu'):
        """初始化XGBoost推荐模型，集成词向量增强文本特征"""
        self.logger = Logger('xgboost_recommendation.log').get_logger()
        self.data_paths = data_paths
        self.device = device

        # 数据与特征相关
        self.train_data = None
        self.test_data = None
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        self.cat_encoders = {}
        self.book_name_map = {}

        # 词向量相关
        self.w2v_model = None  # Word2Vec模型
        self.book_text_vectors = {}  # 书籍文本向量映射
        self.stopwords = self._load_stopwords()  # 停用词列表
        self.text_vec_dim = 100  # 词向量维度

        # 模型相关
        self.model = None
        self.scaler = MinMaxScaler()
        self.best_params = None
        self.metrics = {}

        # 加载书籍名称映射
        if 'books' in data_paths and os.path.exists(data_paths['books']):
            try:
                books_df = pd.read_csv(data_paths['books'])
                self.book_name_map = dict(zip(books_df['book_id'], books_df['题名']))
                self.logger.info(f"成功加载{len(self.book_name_map)}本图书的名称映射")
            except Exception as e:
                self.logger.warning(f"加载书籍名称映射失败: {str(e)}")

    def _load_stopwords(self):
        """加载中文停用词（需提前准备stopwords.txt）"""
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        except:
            self.logger.warning("未找到停用词文件，使用空停用词列表")
            return []

    def _preprocess_text(self, text):
        """文本预处理：分词、去停用词、去除特殊字符"""
        if not text or pd.isna(text):
            return []
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', str(text))
        # 中文分词
        words = jieba.lcut(text)
        # 去停用词和单字
        return [word for word in words if word not in self.stopwords and len(word) > 1]

    def train_word2vec(self, books_df):
        """训练Word2Vec模型，生成书籍文本向量"""
        # 预处理所有书籍标题
        texts = books_df['题名'].apply(self._preprocess_text).tolist()
        # 训练Word2Vec
        self.w2v_model = Word2Vec(
            sentences=texts,
            vector_size=self.text_vec_dim,
            window=5,
            min_count=2,
            workers=4,
            seed=42
        )
        self.logger.info(f"Word2Vec模型训练完成，词汇量：{len(self.w2v_model.wv)}")

        # 生成每本书的文本向量（词向量平均值）
        for idx, row in books_df.iterrows():
            book_id = row['book_id']
            text = self._preprocess_text(row['题名'])
            if not text:
                self.book_text_vectors[book_id] = np.zeros(self.text_vec_dim)
                continue
            # 计算词向量平均值
            vecs = [self.w2v_model.wv[word] for word in text if word in self.w2v_model.wv]
            if vecs:
                self.book_text_vectors[book_id] = np.mean(vecs, axis=0)
            else:
                self.book_text_vectors[book_id] = np.zeros(self.text_vec_dim)
        self.logger.info(f"生成{len(self.book_text_vectors)}本图书的文本向量")

    def load_and_preprocess_data(self, test_size=0.2, neg_sample_ratio=3):
        """加载并预处理数据，集成词向量特征"""
        self.logger.info("开始XGBoost数据预处理...")
        start_time = time.time()

        # 加载增强借阅数据
        enhanced_borrows = pd.read_csv(
            "../../data/merged_borrows_enhanced.csv",
            encoding='utf-8',
            parse_dates=["借阅时间", "还书时间", "续借时间"]
        )

        # 加载书籍数据并训练词向量
        books_df = pd.read_csv(self.data_paths['books'])
        self.train_word2vec(books_df)  # 训练Word2Vec并生成书籍文本向量

        # 1. 特征提取：用户特征、物品特征（含文本向量）、交互特征
        features = self._build_features(enhanced_borrows)

        # 2. 构建正负样本
        train_df = self._build_pos_neg_samples(
            enhanced_borrows,
            neg_sample_ratio=neg_sample_ratio
        )

        # 3. 特征编码与归一化（核心修改处）
        train_df = self._encode_features(train_df)

        # 4. 划分训练集和测试集
        self.train_data, self.test_data = train_test_split(
            train_df,
            test_size=test_size,
            random_state=42,
            stratify=train_df['label']
        )

        self.logger.info(f"数据预处理完成 - 训练集: {len(self.train_data)}条, 测试集: {len(self.test_data)}条")
        self.logger.info(f"正负样本比例 - 训练集: {sum(self.train_data['label']) / len(self.train_data):.2f}, "
                         f"测试集: {sum(self.test_data['label']) / len(self.test_data):.2f}")
        self.logger.info(f"预处理耗时: {time.time() - start_time:.2f}秒")

    def _build_features(self, data):
        """构建用户、物品（含文本向量）、交互三类特征"""
        # 用户特征（仅数值型特征）
        user_features = data.groupby('user_id').agg({
            'book_id': 'nunique',  # 用户借阅书籍数量
            '交互权重': 'sum',       # 用户总交互权重
            'dept_book_preference': 'mean',  # 用户院系-书籍偏好均值
            '续借次数': ['sum', 'mean']  # 续借统计
        }).reset_index()
        user_features.columns = ['user_id'] + [f'user_{c[0]}_{c[1]}' if c[1] else f'user_{c[0]}'
                                               for c in user_features.columns[1:]]

        # 物品特征（含分类特征和数值特征，后续会分离处理）
        book_features = data.groupby('book_id').agg({
            'user_id': 'nunique',  # 借阅用户数量
            '二级分类': 'first',     # 书籍二级分类（分类特征）
            '续借次数': ['sum', 'mean']  # 书籍续借统计（数值特征）
        }).reset_index()
        book_features.columns = ['book_id'] + [f'book_{c[0]}_{c[1]}' if c[1] else f'book_{c[0]}'
                                               for c in book_features.columns[1:]]

        # 合并书籍文本向量特征（数值型）
        text_vec_cols = [f'book_text_vec_{i}' for i in range(self.text_vec_dim)]
        text_vec_df = pd.DataFrame.from_dict(
            self.book_text_vectors,
            orient='index',
            columns=text_vec_cols
        ).reset_index().rename(columns={'index': 'book_id'})
        book_features = book_features.merge(text_vec_df, on='book_id', how='left')

        return {
            'user_features': user_features,
            'book_features': book_features
        }

    def _build_pos_neg_samples(self, data, neg_sample_ratio=3):
        """构建正负样本，正样本为真实借阅，负样本为未借阅书籍"""
        # 正样本
        pos_samples = data[['user_id', 'book_id']].drop_duplicates()
        pos_samples['label'] = 1

        # 负样本
        all_books = set(data['book_id'].unique())
        neg_samples = []

        for user_id, group in tqdm(data.groupby('user_id'), desc="构建负样本"):
            user_books = set(group['book_id'])
            candidate_books = list(all_books - user_books)
            if not candidate_books:
                continue

            sample_size = min(len(user_books) * neg_sample_ratio, len(candidate_books))
            neg_books = random.sample(candidate_books, sample_size)
            neg_samples.extend([(user_id, book, 0) for book in neg_books])

        neg_samples = pd.DataFrame(neg_samples, columns=['user_id', 'book_id', 'label'])

        # 合并正负样本并关联特征
        samples = pd.concat([pos_samples, neg_samples], ignore_index=True)
        # 关联用户特征（全数值）
        samples = samples.merge(self._build_features(data)['user_features'], on='user_id', how='left')
        # 关联物品特征（含分类+数值）
        samples = samples.merge(self._build_features(data)['book_features'], on='book_id', how='left')

        return samples

    def _encode_features(self, data):
        """编码类别特征，归一化数值特征（核心修复：严格区分类别/数值列）"""
        # 1. 编码用户ID和书籍ID（标识类特征）
        self.user_encoder.fit(data['user_id'].unique())
        data['user_id_encoded'] = self.user_encoder.transform(data['user_id'])

        self.book_encoder.fit(data['book_id'].unique())
        data['book_id_encoded'] = self.book_encoder.transform(data['book_id'])

        # 2. 识别并编码所有类别特征（含"分类"关键词的列）
        cat_cols = [col for col in data.columns if '分类' in col]  # 仅捕获分类相关列（如book_二级分类_first）
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = data[col].fillna('未知')  # 填充缺失分类值为"未知"
            data[f'{col}_encoded'] = le.fit_transform(data[col])  # 生成编码后的数值列
            self.cat_encoders[col] = le  # 保存编码器用于后续推荐

        # 3. 识别并归一化所有数值特征（排除类别列、ID列、标签列）
        num_cols = [col for col in data.columns if
                    # 属于用户/物品特征（含文本向量）
                    ('user_' in col or 'book_' in col) and
                    # 不是编码后的类别列
                    '_encoded' not in col and
                    # 排除原始ID和标签
                    col not in ['user_id', 'book_id', 'label'] and
                    # 关键：排除原始分类列（避免字符串混入）
                    col not in cat_cols]

        # 填充数值列缺失值并归一化
        data[num_cols] = self.scaler.fit_transform(data[num_cols].fillna(0))

        # 4. 筛选最终输入模型的特征列（仅用编码后的类别列+归一化的数值列）
        feature_cols = ['user_id_encoded', 'book_id_encoded'] + \
                       [f'{col}_encoded' for col in cat_cols] + \
                       num_cols

        return data[feature_cols + ['label']]

    def optimize_hyperparameters(self, n_trials=20):
        """使用Optuna优化XGBoost超参数（需安装optuna：pip install optuna）"""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            self.logger.error("Optuna未安装，无法进行超参数优化，请执行`pip install optuna`")
            return None

        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
                'random_state': 42,
                'verbosity': 0
            }

            X_sample, _, y_sample, _ = train_test_split(
                self.train_data.drop('label', axis=1),
                self.train_data['label'],
                test_size=0.7,
                random_state=42
            )

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X_sample):
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(
                    X_sample.iloc[train_idx], y_sample.iloc[train_idx],
                    eval_set=[(X_sample.iloc[val_idx], y_sample.iloc[val_idx])],
                    early_stopping_rounds=10,
                    verbose=False
                )
                val_pred = xgb_model.predict(X_sample.iloc[val_idx])
                scores.append(f1_score(y_sample.iloc[val_idx], val_pred))
            return np.mean(scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(study_name='xgb_book_recommender', sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.logger.info(f"最佳超参数: {self.best_params}")
        return self.best_params

    def train(self, params=None, n_splits=5, optimize_hyperparams=False, n_trials=20):
        """训练XGBoost模型，支持交叉验证和超参数优化"""
        if self.train_data is None:
            raise ValueError("请先调用load_and_preprocess_data加载数据")

        self.logger.info("开始训练XGBoost模型...")
        start_time = time.time()

        feature_cols = [col for col in self.train_data.columns if col != 'label']
        X_train, y_train = self.train_data[feature_cols], self.train_data['label']
        X_test, y_test = self.test_data[feature_cols], self.test_data['label']

        # 超参数优化
        if optimize_hyperparams:
            self.optimize_hyperparameters(n_trials=n_trials)

        # 加载最佳参数或默认参数
        if params is None:
            params = self.best_params or {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),  # 平衡正负样本
                'random_state': 42,
                'verbosity': 1
            }

        # 交叉验证（评估模型稳定性）
        if n_splits > 1:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                xgb_model = xgb.XGBClassifier(**params)
                xgb_model.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    early_stopping_rounds=20,
                    verbose=False
                )

                val_pred = xgb_model.predict(X_train.iloc[val_idx])
                val_f1 = f1_score(y_train.iloc[val_idx], val_pred)
                cv_scores.append(val_f1)
                self.logger.info(f"Fold {fold + 1}/{n_splits} - 验证F1: {val_f1:.4f}")

            self.logger.info(f"交叉验证平均F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # 训练最终模型（用全量训练集）
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=True
        )

        # 多维度评估测试集性能
        self._evaluate(X_test, y_test)

        total_time = time.time() - start_time
        self.logger.info(f"XGBoost训练完成，总耗时: {total_time:.2f}秒")
        return self.model

    def _evaluate(self, X, y_true, top_n=10):
        """多维度评估模型性能，含Top-N推荐指标"""
        y_pred_proba = self.model.predict_proba(X)[:, 1]  # 正样本概率
        y_pred = self.model.predict(X)  # 类别预测

        # 1. 二分类基础指标
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1'] = f1_score(y_true, y_pred)
        self.metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

        # 2. Top-N推荐指标（模拟推荐场景）
        top_indices = np.argsort(-y_pred_proba)[:top_n]  # 取概率最高的N个样本
        y_true_top = y_true.iloc[top_indices]  # 真实标签
        self.metrics[f'precision@{top_n}'] = precision_score(y_true_top, [1]*top_n)  # 预测均为正样本
        self.metrics[f'recall@{top_n}'] = recall_score(y_true_top, [1]*top_n)

        # 打印评估结果
        self.logger.info("\n测试集评估指标:")
        for metric, value in self.metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        print(f"F1: {self.metrics['f1']:.4f} | AUC: {self.metrics['auc']:.4f} | "
              f"Precision@10: {self.metrics['precision@10']:.4f} | Recall@10: {self.metrics['recall@10']:.4f}")

    def recommend(self, user_id, top_n=10):
        """为用户推荐Top-N书籍，返回书籍ID和名称"""
        if self.model is None:
            raise ValueError("请先训练模型")

        # 1. 筛选候选书籍（排除用户已借阅的）
        all_books = set(self.book_encoder.classes_)  # 所有书籍ID
        # 从训练数据中获取用户已借阅书籍
        user_borrowed = set(self.train_data[self.train_data['user_id_encoded'] ==
                                            self.user_encoder.transform([user_id])[0]]['book_id_encoded'])
        # 转换回原始book_id（编码器逆变换）
        user_borrowed_raw = self.book_encoder.inverse_transform(list(user_borrowed))
        candidates_raw = list(all_books - set(user_borrowed_raw))  # 候选书籍原始ID

        if not candidates_raw:
            self.logger.warning(f"用户{user_id}已借阅所有书籍，无推荐候选")
            return []

        # 2. 为候选书籍构建特征
        candidate_df = self._build_candidate_features(user_id, candidates_raw)
        feature_cols = [col for col in candidate_df.columns if col != 'book_id']  # 排除原始book_id

        # 3. 预测候选书籍的推荐分数
        candidate_df['score'] = self.model.predict_proba(candidate_df[feature_cols])[:, 1]
        # 按分数降序取Top-N
        top_candidates = candidate_df.sort_values('score', ascending=False).head(top_n)

        # 4. 整理推荐结果（原始book_id + 书名）
        top_book_ids = top_candidates['book_id'].tolist()
        top_book_names = [self.book_name_map.get(book_id, f"未知书籍_{book_id}") for book_id in top_book_ids]

        return list(zip(top_book_ids, top_book_names))

    def _build_candidate_features(self, user_id, candidates_raw):
        """为推荐候选集构建特征（与训练时特征逻辑一致）"""
        # 1. 构建基础候选DataFrame
        candidate_df = pd.DataFrame({
            'user_id': [user_id] * len(candidates_raw),
            'book_id': candidates_raw
        })

        # 2. 关联用户特征（复用训练时的特征逻辑）
        # 从训练数据中提取用户特征（取第一条该用户的特征，因用户特征是聚合后的固定值）
        user_feat_cols = [col for col in self._build_features(self.train_data)['user_features'].columns
                          if col != 'user_id']
        user_feat = self.train_data[self.train_data['user_id_encoded'] ==
                                    self.user_encoder.transform([user_id])[0]][user_feat_cols].iloc[0]
        # 赋值给所有候选行
        for col in user_feat_cols:
            candidate_df[col] = user_feat[col]

        # 3. 关联物品特征（含分类特征和文本向量）
        book_features = self._build_features(self.train_data)['book_features']
        candidate_df = candidate_df.merge(book_features, on='book_id', how='left')

        # 4. 特征编码（与训练时一致）
        # 编码ID
        candidate_df['user_id_encoded'] = self.user_encoder.transform(candidate_df['user_id'])
        candidate_df['book_id_encoded'] = self.book_encoder.transform(candidate_df['book_id'])

        # 编码分类特征
        for col in self.cat_encoders:
            le = self.cat_encoders[col]
            candidate_df[col] = candidate_df[col].fillna('未知')
            # 处理训练时未出现的新分类值
            candidate_df[col] = candidate_df[col].apply(
                lambda x: x if x in le.classes_ else '未知'
            )
            candidate_df[f'{col}_encoded'] = le.transform(candidate_df[col])

        # 归一化数值特征
        num_cols = [col for col in candidate_df.columns if
                    ('user_' in col or 'book_' in col) and
                    '_encoded' not in col and
                    col not in ['user_id', 'book_id'] and
                    col not in self.cat_encoders.keys()]
        candidate_df[num_cols] = self.scaler.transform(candidate_df[num_cols].fillna(0))

        return candidate_df


# 使用示例
if __name__ == "__main__":
    # 数据路径配置（根据实际路径调整）
    data_paths = {
        "books": "../../data/item.csv",  # 需包含book_id和"题名"列
        "borrows": "../../data/inter_preliminary.csv",
        "users": "../../data/user.csv"
    }

    # 初始化模型
    xgb_rec = XGBoostRecommender(data_paths)

    # 加载并预处理数据（负样本比例3:1，可根据数据量调整）
    xgb_rec.load_and_preprocess_data(neg_sample_ratio=3)

    # 训练模型（n_splits=5为5折交叉验证，optimize_hyperparams=True开启超参优化）
    xgb_rec.train(n_splits=5, optimize_hyperparams=False)

    # 为示例用户推荐书籍（取训练集中第一个用户）
    sample_user = self.user_encoder.inverse_transform([self.train_data['user_id_encoded'].iloc[0]])[0]
    recommendations = xgb_rec.recommend(sample_user, top_n=10)
    print(f"\n为用户{sample_user}推荐的书籍:")
    for book_id, book_name in recommendations:
        print(f"  {book_id} - {book_name}")