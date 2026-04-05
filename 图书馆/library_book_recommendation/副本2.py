def load_and_preprocess_data(self, test_size=0.2, random_state=42):
    """
    流程：清洗原始数据 → 合并分割 → 训练集特征工程 → 用训练集参数处理验证集
    """
    self.logger.info("开始数据预处理（先分割后特征工程）...")
    start_time = time.time()

    # ---------------------- 1. 清洗原始数据并保存 ----------------------
    # 清洗图书数据
    books_clean = clean_book_secondary_category(self.data_paths['books'])
    cleaned_books_path = "data/cleaned_books.csv"
    books_clean.to_csv(cleaned_books_path, index=False, encoding='utf-8')
    self.logger.info(f"图书数据清洗完成: {len(books_clean)}条 -> {cleaned_books_path}")

    # 清洗借阅数据（保留日期列，用于后续特征工程）
    borrows_clean = clean_inter_data(self.data_paths['borrows'])
    cleaned_borrows_path = "data/cleaned_borrows.csv"
    borrows_clean.to_csv(cleaned_borrows_path, index=False, encoding='utf-8')
    self.logger.info(f"借阅数据清洗完成: {len(borrows_clean)}条 -> {cleaned_borrows_path}")

    # 保存用户原始数据
    user_df = pd.read_csv(self.data_paths['users'], encoding='utf-8')
    cleaned_user_path = "data/cleaned_user.csv"
    user_df.to_csv(cleaned_user_path, index=False, encoding='utf-8')
    self.logger.info(f"用户数据保存完成: {len(user_df)}条 -> {cleaned_user_path}")

    # ---------------------- 2. 合并分割清洗后的数据（生成训练/验证/测试集） ----------------------
    merged_path = "data/merged_data.csv"
    train_path = "data/train_data.csv"
    valid_path = "data/valid_data.csv"
    test_path = "data/test_data.csv"

    merge_and_split_data(
        inter_path=cleaned_borrows_path,
        item_path=cleaned_books_path,
        user_path=cleaned_user_path,
        merged_path=merged_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        encoding='utf-8',
        split_ratios=(1 - test_size - 0.1, 0.1, test_size),
        random_state=random_state
    )
    self.logger.info(f"数据合并分割完成: 训练集{len(pd.read_csv(train_path))}条，验证集{len(pd.read_csv(valid_path))}条")

    # ---------------------- 3. 训练集特征工程（生成增强特征） ----------------------
    train_data = pd.read_csv(train_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
    train_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
    train_users = pd.read_csv(cleaned_user_path, encoding='utf-8')

    # 时间特征提取
    train_inter_enhanced = extract_time_features(train_data.copy())
    # 交互权重构建
    train_inter_enhanced = build_interaction_weight(train_inter_enhanced)
    # 分类特征编码（留存编码器用于验证/测试集）
    train_users_encoded, train_books_encoded = encode_categorical_features(train_users.copy(), train_books.copy())
    self.user_encoder = train_users_encoded['DEPT'].astype('category').cat.categories
    self.book_cat_encoder = train_books_encoded['二级分类'].astype('category').cat.categories
    # 合并院系特征到交互表
    train_inter_enhanced = merge_dept_features_to_inter(train_inter_enhanced.copy(), train_users.copy(), train_books.copy())
    # 增强用户表（基于训练集统计）
    train_users_enhanced = add_user_dept_features(train_users_encoded.copy(), train_inter_enhanced.copy(), train_books.copy())

    # 保存训练集增强结果
    train_inter_enhanced.to_csv("data/cleaned_borrows_enhanced.csv", index=False, encoding='utf-8')
    train_books_encoded.to_csv("data/cleaned_books_encoded.csv", index=False, encoding='utf-8')
    train_users_enhanced.to_csv("data/cleaned_user_enhanced.csv", index=False, encoding='utf-8')
    self.logger.info(f"训练集特征工程完成: {len(train_inter_enhanced)}条 -> cleaned_borrows_enhanced.csv")

    # ---------------------- 4. 验证集特征工程（复用训练集编码/统计逻辑） ----------------------
    valid_data = pd.read_csv(valid_path, encoding='utf-8', parse_dates=["借阅时间", "还书时间", "续借时间"])
    valid_books = pd.read_csv(cleaned_books_path, encoding='utf-8')
    valid_users = pd.read_csv(cleaned_user_path, encoding='utf-8')

    # 时间特征提取（同训练集逻辑）
    valid_inter_enhanced = extract_time_features(valid_data.copy())
    # 交互权重构建（同训练集逻辑）
    valid_inter_enhanced = build_interaction_weight(valid_inter_enhanced)
    # 分类特征编码（使用训练集编码器）
    valid_users_encoded = valid_users.copy().rename(columns={"借阅人": "user_id"})
    valid_users_encoded['DEPT_encoded'] = valid_users_encoded['DEPT'].astype('category').cat.set_categories(self.user_encoder).cat.codes
    valid_books_encoded = pd.get_dummies(valid_books, columns=['一级分类'], prefix='book_cat1')
    valid_books_encoded['二级分类_encoded'] = valid_books_encoded['二级分类'].astype('category').cat.set_categories(self.book_cat_encoder).cat.codes
    # 合并院系特征（同训练集逻辑）
    valid_inter_enhanced = merge_dept_features_to_inter(valid_inter_enhanced.copy(), valid_users.copy(), valid_books.copy())
    # 增强用户表（复用训练集统计结果）
    valid_users_enhanced = add_user_dept_features(valid_users_encoded.copy(), valid_inter_enhanced.copy(), valid_books.copy())

    # 保存验证集增强结果（可选，若需复用）
    valid_inter_enhanced.to_csv("data/valid_borrows_enhanced.csv", index=False, encoding='utf-8')
    valid_books_encoded.to_csv("data/valid_books_encoded.csv", index=False, encoding='utf-8')
    valid_users_enhanced.to_csv("data/valid_user_enhanced.csv", index=False, encoding='utf-8')
    self.logger.info(f"验证集特征处理完成: {len(valid_inter_enhanced)}条 -> valid_borrows_enhanced.csv")

    # ---------------------- 5. 加载最终训练/验证集（含增强特征） ----------------------
    self.train_data = pd.read_csv("data/cleaned_borrows_enhanced.csv", encoding='utf-8', parse_dates=["借阅时间"])
    self.valid_data = pd.read_csv("data/valid_borrows_enhanced.csv", encoding='utf-8', parse_dates=["借阅时间"])
    self.logger.info(f"预处理完成 - 训练集: {len(self.train_data)}条, 验证集: {len(self.valid_data)}条")
    self.logger.info(f"总耗时: {time.time() - start_time:.2f}秒")