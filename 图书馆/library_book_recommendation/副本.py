from 图书馆.library_book_recommendation.data_preprocessing.feature_engineering import (
    extract_time_features,
    build_interaction_weight,
    encode_categorical_features,
    merge_dept_features_to_inter,
    add_user_dept_features
)
def load_and_preprocess_data(self, test_size=0.2, random_state=42):
    """
    完整数据预处理流程：清洗原始数据 → 特征工程 → 合并分割
    """
    self.logger.info("开始数据预处理...")
    start_time = time.time()

    # ---------------------- 1. 清洗原始数据并保存 ----------------------
    # 清洗图书数据
    books_clean = clean_book_secondary_category(self.data_paths['books'])
    cleaned_books_path = "data/cleaned_books.csv"
    books_clean.to_csv(cleaned_books_path, index=False, encoding='utf-8')
    self.logger.info(f"图书数据清洗完成: {len(books_clean)}条 -> {cleaned_books_path}")

    # 清洗借阅数据
    borrows_clean = clean_inter_data(self.data_paths['borrows'])
    cleaned_borrows_path = "data/cleaned_borrows.csv"
    borrows_clean.to_csv(cleaned_borrows_path, index=False, encoding='utf-8')
    self.logger.info(f"借阅数据清洗完成: {len(borrows_clean)}条 -> {cleaned_borrows_path}")

    # ---------------------- 2. 执行特征工程（核心新增步骤） ----------------------
    # 读取清洗后的数据
    borrows_clean = pd.read_csv(cleaned_borrows_path, encoding='utf-8',
                                parse_dates=["借阅时间", "还书时间", "续借时间"])
    books_clean = pd.read_csv(cleaned_books_path, encoding='utf-8')
    user_df = pd.read_csv(self.data_paths['users'], encoding='utf-8')  # 读取原始用户数据

    # ① 时间特征（借阅月份、是否学期内）
    borrows_clean = extract_time_features(borrows_clean)
    # ② 交互权重（续借次数→兴趣强度）
    borrows_clean = build_interaction_weight(borrows_clean)
    # ③ 分类特征编码（用户院系、图书分类）
    user_df_encoded, books_clean_encoded = encode_categorical_features(user_df.copy(), books_clean.copy())
    # ④ 合并院系-图书/分类偏好到交互表
    borrows_enhanced = merge_dept_features_to_inter(borrows_clean.copy(), user_df.copy(), books_clean.copy())
    # ⑤ 增强用户表（院系平均续借、分类偏好等）
    user_enhanced = add_user_dept_features(user_df_encoded, borrows_clean, books_clean.copy())

    # 保存增强后的数据（可选，用于 debug 或后续复用）
    enhanced_borrows_path = "data/cleaned_borrows_enhanced.csv"
    enhanced_books_path = "data/cleaned_books_encoded.csv"
    enhanced_user_path = "data/cleaned_user_enhanced.csv"
    borrows_enhanced.to_csv(enhanced_borrows_path, index=False, encoding='utf-8')
    books_clean_encoded.to_csv(enhanced_books_path, index=False, encoding='utf-8')
    user_enhanced.to_csv(enhanced_user_path, index=False, encoding='utf-8')
    self.logger.info(f"特征工程完成: 增强后借阅数据 {len(borrows_enhanced)}条 -> {enhanced_borrows_path}")

    # ---------------------- 3. 合并+分割增强后的数据 ----------------------
    merged_path = "data/merged_data.csv"
    train_path = "data/train_data.csv"
    valid_path = "data/valid_data.csv"
    test_path = "data/test_data.csv"

    # 调用合并分割函数（传入**增强后**的数据路径）
    merge_and_split_data(
        inter_path=enhanced_borrows_path,  # 增强后的借阅数据
        item_path=enhanced_books_path,  # 编码后的图书数据
        user_path=enhanced_user_path,  # 增强后的用户数据
        merged_path=merged_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        encoding='utf-8',
        split_ratios=(1 - test_size - 0.1, 0.1, test_size),  # 训练:验证:测试比例
        random_state=random_state
    )

    # ---------------------- 4. 加载分割后的数据 ----------------------
    self.train_data = pd.read_csv(train_path, encoding='utf-8')
    self.valid_data = pd.read_csv(valid_path, encoding='utf-8')
    self.logger.info(f"数据分割完成 - 训练集: {len(self.train_data)}条, 验证集: {len(self.valid_data)}条")
    self.logger.info(f"数据预处理总耗时: {time.time() - start_time:.2f}秒")