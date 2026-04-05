import pandas as pd


def find_existing_user_ids(csv_path, target_user_ids):
    """
    从submission.csv中筛选出存在于目标用户ID列表中的user_id

    参数:
        csv_path: submission.csv文件路径
        target_user_ids: 待查找的用户ID列表（整数或字符串类型）
    """
    # 1. 读取submission.csv
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"文件读取错误: {e}")
        return

    # 2. 检查CSV是否包含user_id列
    if 'user_id' not in df.columns:
        print("错误：submission.csv中未找到'user_id'列，请检查文件格式")
        return

    # 3. 统一数据类型（将CSV中的user_id和目标列表都转为字符串，避免类型差异）
    df['user_id'] = df['user_id'].astype(str)
    target_user_ids_str = [str(id) for id in target_user_ids]

    # 4. 筛选存在于CSV中的目标用户ID
    csv_user_set = set(df['user_id'].unique())  # CSV中所有不重复的user_id
    existing_ids = [user_id for user_id in target_user_ids_str if user_id in csv_user_set]

    # 5. 输出结果
    print("=" * 60)
    print(f"=== submission.csv用户ID匹配结果 ===")
    print(f"目标用户ID总数: {len(target_user_ids)}")
    print(f"存在于submission.csv中的用户ID总数: {len(existing_ids)}")
    print("\n存在的用户ID（排序后）:")
    if existing_ids:
        # 按数值排序（确保整数ID顺序正确）
        existing_ids_sorted = sorted(existing_ids, key=lambda x: int(x) if x.isdigit() else x)
        print(", ".join(existing_ids_sorted))
    else:
        print("未找到任何匹配的用户ID")
    print("=" * 60)


# 示例用法
if __name__ == "__main__":
    # 1. 配置文件路径（请根据实际路径修改）
    submission_path = "submission.csv"

    # 2. 待查找的目标用户ID列表（用户提供的列表）
    target_user_ids = [
        16, 17, 24, 50, 51, 56, 57, 65, 69, 76, 95, 125, 127, 130, 132, 143, 148, 153, 155, 162, 177, 180, 181, 183, 196, 231, 233, 235, 238, 240, 248, 250, 252, 253, 254, 255, 256, 265, 267, 279, 292, 293, 304, 320, 335, 369, 377, 385, 395, 406, 418, 421, 431, 444, 470, 493, 497, 505, 507, 528, 543, 564, 568, 574, 577, 578, 590, 606, 612, 618, 632, 648, 696, 698, 703, 717, 728, 730, 731, 732, 750, 752, 757, 764, 766, 770, 774, 775, 778, 784, 787, 851, 865, 868, 874, 899, 900, 905, 935, 940, 945, 964, 969, 978, 981, 994, 1007, 1023, 1024, 1069, 1076, 1084, 1144, 1157, 1184, 1190, 1218, 1232, 1240, 1245, 1258, 1268, 1291, 1302, 1305, 1326, 1330, 1336, 1366, 1374, 1380, 1382, 1391, 1392, 1395, 1396, 1415, 1427, 1442, 1448
    ]

    # 3. 执行查找
    find_existing_user_ids(submission_path, target_user_ids)