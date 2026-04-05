def calculate_precision_recall(model, valid_inter):
    """
    计算精确率（P）和召回率（R）
    适配修改后的UserCF类：从模型内部获取user_idx/book_idx/book_list，无需外部传入
    参数：
        model: 训练好的UserCF模型（需包含user_idx、book_idx、book_list属性，predict方法仅需user_id）
        valid_inter: 验证集交互数据（DataFrame，含user_id、book_id）
    返回：
        precision: 精确率
        recall: 召回率
    """
    # 构建用户-真实借阅图书字典（从验证集提取）
    user_true_books = valid_inter.groupby('user_id')['book_id'].apply(set).to_dict()

    tp = 0  # 推荐且用户实际借阅（True Positive）
    fp = 0  # 推荐但用户未借阅（False Positive）
    fn = 0  # 用户借阅但未推荐（False Negative）

    # 遍历验证集中的用户，评估模型预测效果
    for user_id in user_true_books.keys():
        # 跳过训练集中未出现的用户（模型无法预测新用户）
        if user_id not in model.user_idx:
            fn += len(user_true_books[user_id])  # 全部视为漏推荐
            continue

        # 模型预测：仅传入user_id（内部用model.user_idx/book_idx/book_list处理）
        pred_book = model.predict(user_id)
        true_books = user_true_books[user_id]

        # 处理模型无法预测的情况（理论上已通过user_id in model.user_idx过滤，此处为兜底）
        if pred_book is None:
            fn += len(true_books)
            continue

        # 统计TP、FP、FN（保持原逻辑：用户可能借多本，仅推荐1本）
        if pred_book in true_books:
            tp += 1  # 推荐命中
        else:
            fp += 1  # 推荐未命中
        # FN = 真实借阅数 - 命中数（命中1本则漏推荐len-1本，未命中则漏推荐全部）
        fn += len(true_books) - (1 if pred_book in true_books else 0)

    # 计算精确率和召回率（避免除以0错误）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def calculate_f1(precision, recall):
    """计算F1 Score（精确率和召回率的调和平均数）{insert\_element\_15\_}"""
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1