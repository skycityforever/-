import pandas as pd
from collections import defaultdict


def format_datetime(dt_series):
    """将时间列格式化到秒（处理可能的毫秒/时区问题）"""
    dt = pd.to_datetime(dt_series, errors='coerce')
    return dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')


def safe_book_id_to_int(book_id_str):
    """安全转换book_id为整数（失败返回None）"""
    try:
        return int(book_id_str)
    except (ValueError, TypeError):
        return None  # 非数字book_id无法计算差异


def calculate_global_record_diff(file1_path, file2_path, time_col='借阅时间'):
    """全局统计两个文件的记录差异，含书籍明显变化统计"""
    try:
        df1 = pd.read_csv(
            file1_path,
            usecols=['user_id', 'book_id', time_col],
            encoding='utf-8'
        )
        df2 = pd.read_csv(
            file2_path,
            usecols=['user_id', 'book_id', time_col],
            encoding='utf-8'
        )
    except Exception as e:
        print(f"文件读取错误: {e}")
        return

    for df in [df1, df2]:
        df['user_id'] = df['user_id'].astype(str)
        df['book_id'] = df['book_id'].astype(str)
        df['time_key'] = format_datetime(df[time_col])
        df['unique_key'] = df['user_id'] + '|' + df['time_key']

    key_to_book1_str = dict(df1[['unique_key', 'book_id']].drop_duplicates('unique_key', keep='last').values)
    key_to_book2_str = dict(df2[['unique_key', 'book_id']].drop_duplicates('unique_key', keep='last').values)
    all_keys = set(key_to_book1_str.keys()).union(set(key_to_book2_str.keys()))

    total_records = len(all_keys)
    new_records = 0
    lost_records = 0
    book_changed_records = 0
    significant_changed_records = 0
    invalid_book_id_records = 0

    for key in all_keys:
        has1 = key in key_to_book1_str
        has2 = key in key_to_book2_str

        if has1 and has2:
            book1_str = key_to_book1_str[key]
            book2_str = key_to_book2_str[key]
            if book1_str != book2_str:
                book_changed_records += 1
                book1_int = safe_book_id_to_int(book1_str)
                book2_int = safe_book_id_to_int(book2_str)
                if book1_int is not None and book2_int is not None:
                    if abs(book1_int - book2_int) > 8000:
                        significant_changed_records += 1
                else:
                    invalid_book_id_records += 1
        elif has1 and not has2:
            lost_records += 1
        elif not has1 and has2:
            new_records += 1

    print("=" * 60)
    print("=== 全局记录差异统计（基于user_id+借阅时间精确到秒） ===")
    print(f"两个文件中所有唯一记录总数: {total_records}")
    print(f"仅在文件2中存在的新增记录: {new_records} ({new_records / total_records * 100:.2f}%)")
    print(f"仅在文件1中存在的消失记录: {lost_records} ({lost_records / total_records * 100:.2f}%)")
    print(f"唯一键相同但book_id变化的记录: {book_changed_records} ({book_changed_records / total_records * 100:.2f}%)")
    print("\n--- 书籍编号明显变化统计（定义：两个book_id的数值差绝对值 > 8000） ---")
    print(f"明显变化的记录数: {significant_changed_records}")
    print(f"   占所有变化记录的比例: {significant_changed_records / book_changed_records * 100:.2f}%"
          if book_changed_records > 0 else "   无变化记录，无法计算比例")
    print(f"无法计算差异的记录数（非数字book_id）: {invalid_book_id_records}")
    print(f"总变化记录数（新增+消失+book_id变化）: {new_records + lost_records + book_changed_records}")
    print("=" * 60)

    key_to_book1_int = {k: safe_book_id_to_int(v) for k, v in key_to_book1_str.items()}
    key_to_book2_int = {k: safe_book_id_to_int(v) for k, v in key_to_book2_str.items()}
    return (key_to_book1_str, key_to_book2_str,
            key_to_book1_int, key_to_book2_int,
            all_keys, book_changed_records)


def calculate_user_similarity(key_to_book1_str, key_to_book2_str,
                              key_to_book1_int, key_to_book2_int,
                              all_keys, total_global_changed):
    """按用户分组计算相似度及指标，新增90%-99%用户ID统计"""
    user_stats = defaultdict(lambda: {
        'total_records': 0,
        'changed_records': 0,
        'significant_changed_records': 0
    })

    for key in all_keys:
        user_id = key.split('|')[0] if '|' in key else key
        user_stats[user_id]['total_records'] += 1

        has1 = key in key_to_book1_str
        has2 = key in key_to_book2_str
        if has1 and has2 and key_to_book1_str[key] != key_to_book2_str[key]:
            user_stats[user_id]['changed_records'] += 1
            book1_int = key_to_book1_int[key]
            book2_int = key_to_book2_int[key]
            if book1_int is not None and book2_int is not None:
                if abs(book1_int - book2_int) > 8000:
                    user_stats[user_id]['significant_changed_records'] += 1

    all_users = set(user_stats.keys())
    total_users = len(all_users)

    changed_users = 0
    for user in all_users:
        if user_stats[user]['changed_records'] > 0:
            changed_users += 1
    changed_users_percent = (changed_users / total_users) * 100 if total_users > 0 else 0

    high_confusion_total = 0
    high_confusion_users = 0
    for user in all_users:
        total = user_stats[user]['total_records']
        changed = user_stats[user]['changed_records']
        if total == 0:
            continue
        if changed / total > 0.5:
            high_confusion_total += changed
            high_confusion_users += 1

    significant_changed_users = 0
    total_user_significant_changed = 0
    for user in all_users:
        if user_stats[user]['significant_changed_records'] > 0:
            significant_changed_users += 1
            total_user_significant_changed += user_stats[user]['significant_changed_records']
    significant_users_percent = (significant_changed_users / changed_users) * 100 if changed_users > 0 else 0

    # 调整分档：细化80%-100%为80%-90%、90%-99%、100%
    bin_users = {
        "0-30%": [],
        "30%-50%": [],
        "50%-80%": [],
        "80%-90%": [],  # 新增
        "90%-99%": [],  # 新增
        "100%": []  # 原exact_100_users
    }
    bins = {k: 0 for k in bin_users.keys()}  # 分档计数器

    # 10号用户统计
    target_user = "10"
    target_match_count = 0
    target_total_common = 0
    target_similarity = 0.0

    for user in all_users:
        user_common_keys = []
        for key in all_keys:
            if key.startswith(f"{user}|") and key in key_to_book1_str and key in key_to_book2_str:
                user_common_keys.append(key)

        total_common = len(user_common_keys)
        if total_common == 0:
            similarity = 0.0
        else:
            match_count = sum(
                1 for key in user_common_keys
                if key_to_book1_str[key] == key_to_book2_str[key]
            )
            similarity = match_count / total_common

        # 10号用户数据
        if user == target_user:
            target_match_count = match_count
            target_total_common = total_common
            target_similarity = similarity

        # 分档判断（细化80%-100%区间）
        if similarity < 0.3:
            bins["0-30%"] += 1
            bin_users["0-30%"].append(user)
        elif 0.3 <= similarity < 0.5:
            bins["30%-50%"] += 1
            bin_users["30%-50%"].append(user)
        elif 0.5 <= similarity < 0.8:
            bins["50%-80%"] += 1
            bin_users["50%-80%"].append(user)
        elif 0.8 <= similarity < 0.9:  # 80%-90%
            bins["80%-90%"] += 1
            bin_users["80%-90%"].append(user)
        elif 0.9 <= similarity < 1.0:  # 90%-99%
            bins["90%-99%"] += 1
            bin_users["90%-99%"].append(user)
        else:  # 100%
            bins["100%"] += 1
            bin_users["100%"].append(user)

    # 输出10号用户统计
    print("\n" + "=" * 60)
    print(f"=== 10号用户（user_id={target_user}）的相似度统计 ===")
    if target_user not in all_users:
        print("该用户在两个文件中无任何记录")
    else:
        print(f"1. 同一时间在两个文件中都存在的总记录数: {target_total_common}")
        print(f"2. 同一时间且book_id相同的匹配记录数: {target_match_count}")
        print(f"3. 相似度: {target_similarity:.4f}（{target_similarity * 100:.2f}%）")
    print("=" * 60)

    # 输出用户级变化指标
    print("\n" + "=" * 60)
    print("=== 用户级变化指标统计 ===")
    print(f"1. 发生变化的用户数（至少1条变化记录）: {changed_users} ({changed_users_percent:.2f}%)")
    print(f"2. 混乱程度>50%的用户数: {high_confusion_users}")
    print(f"   这些用户的总变化记录数: {high_confusion_total}")
    print("\n--- 书籍编号明显变化统计（定义：两个book_id的数值差绝对值 > 8000） ---")
    print(f"3. 有明显变化记录的用户数: {significant_changed_users}")
    print(f"   占总变化用户的比例: {significant_users_percent:.2f}%"
          if changed_users > 0 else "   无变化用户，无法计算比例")
    print(f"4. 所有用户的明显变化记录总数: {total_user_significant_changed}")
    print("=" * 60)

    # 输出细化后的分组相似度统计
    print("\n" + "=" * 60)
    print("=== 按用户分组的book_id相似度统计（基于相同时间） ===")
    print(f"总用户数: {total_users}")
    print("说明：相似度 = 同一用户同一时间的book_id匹配数 / 同一用户同一时间的总记录数")
    print("-" * 60)
    for bin_name, count in bins.items():
        percentage = (count / total_users) * 100 if total_users > 0 else 0
        print(f"{bin_name}: {count}人 ({percentage:.2f}%)")
    print("=" * 60)

    # 输出100%没变化的用户ID（对应100%档位）
    sorted_100 = sorted(bin_users["100%"], key=lambda x: int(x) if x.isdigit() else x)
    print("\n" + "=" * 60)
    print(f"=== 100%没变化的用户ID（共{len(sorted_100)}人，排序后） ===")
    print(", ".join(sorted_100))
    print("=" * 60)

    # 新增：输出90%-99%档位的用户ID（排序后）
    sorted_90_99 = sorted(bin_users["90%-99%"], key=lambda x: int(x) if x.isdigit() else x)
    print("\n" + "=" * 60)
    print(f"=== 90%-99% 档位具体用户ID（共{len(sorted_90_99)}人，排序后） ===")
    print(", ".join(sorted_90_99))
    print("=" * 60)

    # 输出80%-90%档位的用户ID（新增，保持完整性）
    sorted_80_90 = sorted(bin_users["80%-90%"], key=lambda x: int(x) if x.isdigit() else x)
    print("\n" + "=" * 60)
    print(f"=== 80%-90% 档位具体用户ID（共{len(sorted_80_90)}人，排序后） ===")
    print(", ".join(sorted_80_90))
    print("=" * 60)

    # 输出0-30%和30%-50%档位用户ID
    sorted_0_30 = sorted(bin_users["0-30%"], key=lambda x: int(x) if x.isdigit() else x)
    sorted_30_50 = sorted(bin_users["30%-50%"], key=lambda x: int(x) if x.isdigit() else x)

    print("\n" + "=" * 60)
    print("=== 0-30% 档位具体用户ID（排序后） ===")
    print(", ".join(sorted_0_30))

    print("\n" + "=" * 60)
    print("=== 30%-50% 档位具体用户ID（排序后） ===")
    print(", ".join(sorted_30_50))
    print("=" * 60)


def compare_two_files(file1_path, file2_path):
    result = calculate_global_record_diff(file1_path, file2_path)
    if result is None:
        return
    key_to_book1_str, key_to_book2_str, key_to_book1_int, key_to_book2_int, all_keys, total_global_changed = result
    calculate_user_similarity(
        key_to_book1_str, key_to_book2_str,
        key_to_book1_int, key_to_book2_int,
        all_keys, total_global_changed
    )


if __name__ == "__main__":
    file1 = "inter_final.csv"
    file2 = "inter_final_选手可见.csv"
    compare_two_files(file1, file2)