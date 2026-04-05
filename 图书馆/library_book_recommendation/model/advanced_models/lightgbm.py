import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class LightGBMRecommender:
    def __init__(self, logger=None):
        """初始化LightGBM推荐模型（专注推荐场景）"""
        # 日志工具（可外部传入，保持系统一致性）
        self.logger = logger if logger else self._default_logger()

        # 数据与特征相关
        self.raw_data = None  # 原始交互数据（用于构建候选集）
        self.train_samples = None  # 训练样本（含正负例）
        self.test_samples = None  # 测试样本
        self.feature_names = None  # 特征列名

        # 编码与归一化工具
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.cat_encoders = {}  # 分类特征编码器（如书籍分类）
        self.scaler = MinMaxScaler()  # 数值特征归一化

        # 模型相关
        self.model = None
        self.best_score = 0.0  # 最佳评估分数

    def _default_logger(self):
        """默认日志工具（无外部传入时使用）"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("LightGBMRecommender")

    def prepare_data(self, inter_data, neg_ratio=3, test_size=0.2):
        """
        准备训练数据（核心步骤：构建正负样本）
        inter_data: 原始交互数据，需包含user_id, item_id（或book_id）
        """
        self.raw_data = inter_data.copy()
        self.logger.info(f"开始准备数据 - 原始交互记录: {len(inter_data)}条")

        # 1. 构建正样本（真实交互）
        pos_samples = inter_data[['user_id', 'book_id']].drop_duplicates()
        pos_samples['label'] = 1  # 正例标记为1

        # 2. 构建负样本（未交互的用户-物品对）
        all_items = set(inter_data['book_id'].unique())
        neg_samples = []

        for user_id, group in tqdm(inter_data.groupby('user_id'), desc="生成负样本"):
            user_items = set(group['book_id'])  # 用户已交互的物品
            candidate_items = list(all_items - user_items)  # 未交互物品作为候选
            if not candidate_items:
                continue

            # 负样本数量 = 正样本数量 × 负正比例（控制样本平衡）
            sample_num = min(len(user_items) * neg_ratio, len(candidate_items))
            neg_items = random.sample(candidate_items, sample_num)
            neg_samples.extend([(user_id, item, 0) for item in neg_items])  # 负例标记为0

        neg_samples = pd.DataFrame(neg_samples, columns=['user_id', 'book_id', 'label'])

        # 3. 合并正负样本并构建特征
        all_samples = pd.concat([pos_samples, neg_samples], ignore_index=True)
        all_samples = self._build_features(all_samples)  # 特征工程

        # 4. 划分训练集和测试集
        self.train_samples, self.test_samples = train_test_split(
            all_samples, test_size=test_size, random_state=42, stratify=all_samples['label']
        )

        # 提取特征列名（排除label）
        self.feature_names = [col for col in all_samples.columns if col != 'label']
        self.logger.info(f"数据准备完成 - 训练样本: {len(self.train_samples)}, 测试样本: {len(self.test_samples)}")

    def _build_features(self, samples):
        """
        构建特征（推荐场景核心特征工程）
        包含：用户特征、物品特征、交互特征
        """
        # ------------ 用户特征 ------------
        user_feats = self.raw_data.groupby('user_id').agg({
            'book_id': 'nunique',  # 用户交互过的物品数量（活跃度）
            '交互权重': 'sum'  # 总交互权重（用户对平台的粘性）
        }).reset_index()
        # 重命名列名（扁平化MultiIndex）
        user_feats.columns = ['user_id'] + [f'user_{c[0]}_{c[1]}'
                                            for c in user_feats.columns[1:]]

        # ------------ 物品特征 ------------
        item_feats = self.raw_data.groupby('book_id').agg({
            'user_id': 'nunique',  # 交互过该物品的用户数（流行度）
            '二级分类': 'first'  # 物品分类（内容特征）
        }).reset_index()
        item_feats.columns = ['book_id'] + [f'item_{c[0]}_{c[1]}'
                                            for c in item_feats.columns[1:]]

        # 关联特征到样本
        samples = samples.merge(user_feats, on='user_id', how='left')
        samples = samples.merge(item_feats, on='book_id', how='left')

        # ------------ 特征编码与归一化 ------------
        # 1. 用户ID和物品ID编码
        self.user_encoder.fit(samples['user_id'].unique())
        samples['user_id_encoded'] = self.user_encoder.transform(samples['user_id'])

        self.item_encoder.fit(samples['book_id'].unique())
        samples['item_id_encoded'] = self.item_encoder.transform(samples['book_id'])

        # 2. 分类特征编码（如物品分类）
        cat_cols = [col for col in samples.columns if '分类' in col]
        for col in cat_cols:
            le = LabelEncoder()
            samples[col] = samples[col].fillna('未知')  # 处理缺失值
            samples[f'{col}_encoded'] = le.fit_transform(samples[col])
            self.cat_encoders[col] = le  # 保存编码器用于推荐

        # 3. 数值特征归一化（加速训练，提升稳定性）
        num_cols = [col for col in samples.columns if
                    ('user_' in col or 'item_' in col) and
                    '_encoded' not in col and
                    col not in ['user_id', 'book_id', 'label']]
        samples[num_cols] = self.scaler.fit_transform(samples[num_cols].fillna(0))

        # 保留有用的列（删除原始ID和未编码的分类特征）
        keep_cols = ['user_id_encoded', 'item_id_encoded', 'label'] + \
                    [f'{col}_encoded' for col in cat_cols] + num_cols
        return samples[keep_cols]

    def train(self, params=None, n_splits=5):
        """
        训练LightGBM模型（推荐场景优化参数）
        params: LightGBM参数（None时使用默认推荐参数）
        n_splits: 交叉验证折数（0表示不使用交叉验证）
        """
        if self.train_samples is None:
            raise ValueError("请先调用prepare_data准备数据")

        # 特征与标签
        X_train = self.train_samples[self.feature_names]
        y_train = self.train_samples['label']
        X_test = self.test_samples[self.feature_names]
        y_test = self.test_samples['label']

        # LightGBM数据集格式（高效存储）
        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

        # 默认参数（针对推荐场景调优）
        if params is None:
            params = {
                'objective': 'binary',  # 二分类：是否交互
                'metric': 'auc',  # 评估指标（推荐场景常用AUC）
                'boosting_type': 'gbdt',
                'num_leaves': 63,  # 叶子数（LightGBM核心参数，比XGBoost更灵活）
                'learning_rate': 0.05,
                'feature_fraction': 0.9,  # 特征采样（防止过拟合）
                'bagging_fraction': 0.8,  # 数据采样
                'bagging_freq': 5,  # 每5轮采样一次
                'verbose': 1,
                # 平衡正负样本（推荐场景负样本通常远多于正样本）
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
                'random_state': 42
            }

        # 交叉验证（评估模型稳定性）
        if n_splits > 1:
            self.logger.info(f"开始{n_splits}折交叉验证...")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
                tr_data = lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
                val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

                # 训练单折模型
                fold_model = lgb.train(
                    params, tr_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )

                # 评估单折性能
                val_pred = fold_model.predict(X_train.iloc[val_idx], num_iteration=fold_model.best_iteration)
                val_pred_label = [1 if p > 0.5 else 0 for p in val_pred]  # 阈值0.5
                fold_f1 = f1_score(y_train.iloc[val_idx], val_pred_label)
                cv_scores.append(fold_f1)
                self.logger.info(f"折{fold + 1} F1: {fold_f1:.4f}")

            self.logger.info(f"交叉验证平均F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # 训练最终模型（在全量训练集上）
        self.logger.info("开始训练最终模型...")
        start_time = time.time()
        self.model = lgb.train(
            params, lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_test],
            early_stopping_rounds=20,
            verbose_eval=10  # 每10轮打印一次日志
        )

        # 评估最终模型
        self._evaluate(X_test, y_test)
        self.logger.info(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒")

    def _evaluate(self, X, y_true):
        """评估模型在推荐场景的核心指标"""
        # 预测概率和类别
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred_label = [1 if p > 0.5 else 0 for p in y_pred_proba]

        # 计算推荐场景关键指标
        precision = precision_score(y_true, y_pred_label)  # 准确率（预测的正例中真实正例比例）
        recall = recall_score(y_true, y_pred_label)  # 召回率（真实正例中被预测的比例）
        f1 = f1_score(y_true, y_pred_label)  # F1分数（综合指标）
        auc = roc_auc_score(y_true, y_pred_proba)  # AUC（排序能力）

        self.logger.info("\n===== 模型评估结果 =====")
        self.logger.info(f"精确率: {precision:.4f}")
        self.logger.info(f"召回率: {recall:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}")
        self.logger.info(f"AUC: {auc:.4f}")
        self.best_score = f1  # 以F1作为最佳分数指标

    def recommend(self, user_id, top_n=10):
        """为指定用户推荐Top-N物品"""
        if self.model is None:
            raise ValueError("请先训练模型")

        # 1. 生成候选集：用户未交互过的物品
        all_items = set(self.item_encoder.classes_)
        user_interacted = set(self.raw_data[self.raw_data['user_id'] == user_id]['book_id'])
        candidates = list(all_items - user_interacted)
        if not candidates:
            self.logger.warning(f"用户{user_id}无未交互物品，无法推荐")
            return []

        # 2. 为候选集构建特征
        candidate_feats = self._build_candidate_features(user_id, candidates)

        # 3. 预测并排序
        candidate_feats['score'] = self.model.predict(
            candidate_feats[self.feature_names],
            num_iteration=self.model.best_iteration
        )
        # 取分数最高的top_n个物品
        top_items = candidate_feats.sort_values('score', ascending=False)['book_id'].head(top_n).tolist()

        return top_items

    def _build_candidate_features(self, user_id, candidates):
        """为推荐候选集构建特征（复用训练时的特征逻辑）"""
        # 构建候选集基础信息
        candidate_df = pd.DataFrame({
            'user_id': [user_id] * len(candidates),
            'book_id': candidates
        })

        # 关联用户特征和物品特征
        user_feats = self.raw_data.groupby('user_id').agg({
            'book_id': 'nunique',
            '交互权重': 'sum'
        }).reset_index()
        user_feats.columns = ['user_id'] + [f'user_{c[0]}_{c[1]}' for c in user_feats.columns[1:]]

        item_feats = self.raw_data.groupby('book_id').agg({
            'user_id': 'nunique',
            '借阅时长': ['mean', 'std'],
            '二级分类': 'first'
        }).reset_index()
        item_feats.columns = ['book_id'] + [f'item_{c[0]}_{c[1]}' for c in item_feats.columns[1:]]

        candidate_df = candidate_df.merge(user_feats, on='user_id', how='left')
        candidate_df = candidate_df.merge(item_feats, on='book_id', how='left')

        # 特征编码（复用训练时的编码器）
        candidate_df['user_id_encoded'] = self.user_encoder.transform(candidate_df['user_id'])
        candidate_df['item_id_encoded'] = self.item_encoder.transform(candidate_df['book_id'])

        # 分类特征编码（处理训练集中未出现的类别）
        cat_cols = [col for col in candidate_df.columns if '分类' in col]
        for col in cat_cols:
            le = self.cat_encoders.get(col, LabelEncoder())
            candidate_df[col] = candidate_df[col].fillna('未知')
            # 未见过的类别映射为'未知'
            candidate_df[col] = candidate_df[col].apply(lambda x: x if x in le.classes_ else '未知')
            candidate_df[f'{col}_encoded'] = le.transform(candidate_df[col])

        # 数值特征归一化
        num_cols = [col for col in candidate_df.columns if
                    ('user_' in col or 'item_' in col) and
                    '_encoded' not in col and
                    col not in ['user_id', 'book_id']]
        candidate_df[num_cols] = self.scaler.transform(candidate_df[num_cols].fillna(0))

        return candidate_df


# 使用示例
if __name__ == "__main__":
    # 1. 准备数据（替换为你的数据路径）
    # 假设数据格式：user_id, book_id, 借阅时长, 交互权重, 二级分类
    data = pd.read_csv("../../data/merged_borrows_enhanced.csv", encoding='utf-8')

    # 2. 初始化模型
    lgb_rec = LightGBMRecommender()

    # 3. 准备训练数据（负样本比例3:1）
    lgb_rec.prepare_data(data, neg_ratio=3)

    # 4. 训练模型（5折交叉验证）
    lgb_rec.train(n_splits=5)

    # 5. 为示例用户推荐
    sample_user = data['user_id'].iloc[0]  # 取第一个用户
    recommendations = lgb_rec.recommend(sample_user, top_n=10)
    print(f"\n为用户{sample_user}推荐的书籍ID: {recommendations}")