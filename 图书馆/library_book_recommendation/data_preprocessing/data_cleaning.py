import pandas as pd
import numpy as np


def clean_user_data(user_path):
    """清洗用户数据：按分组逻辑填充缺失的年级/院系，确保user_id格式为字符串"""
    user_df = pd.read_csv(user_path)

    # 填充缺失的「年级」：优先按「类型」（本科/研究生等）分组取众数，再用整体众数兜底
    user_df['年级'] = user_df.groupby('类型')['年级'].apply(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    user_df['年级'] = user_df['年级'].fillna(user_df['年级'].mode()[0])  # 若分组后仍有缺失，用全体众数

    # 填充缺失的「DEPT（院系）」：优先按「年级+类型」分组取众数，再用整体众数兜底
    user_df['DEPT'] = user_df.groupby(['年级', '类型'])['DEPT'].apply(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    user_df['DEPT'] = user_df['DEPT'].fillna(user_df['DEPT'].mode()[0])

    # 确保「借阅人」（即user_id）为字符串格式（避免后续关联时类型不匹配）
    user_df['借阅人'] = user_df['借阅人'].astype(str)
    return user_df


def clean_inter_data(inter_path):
    """清洗交互数据：适配借阅时间全缺失、续借逻辑自洽"""
    inter_df = pd.read_csv(inter_path)

    # 1. 时间列转为datetime，无效值设为NaT
    inter_df['借阅时间'] = pd.to_datetime(inter_df['借阅时间'], errors='coerce')
    inter_df['还书时间'] = pd.to_datetime(inter_df['还书时间'], errors='coerce')
    inter_df['续借时间'] = pd.to_datetime(inter_df['续借时间'], errors='coerce')

    # 2. 处理借阅时间全缺失：生成“虚拟时间”（按行递增，模拟时间流）
    if inter_df['借阅时间'].isna().all():
        base_time = pd.Timestamp('2020-01-01')  # 起始虚拟时间
        inter_df['借阅时间'] = [base_time + pd.Timedelta(days=i) for i in range(len(inter_df))]
    else:
        # 部分有效时，同用户前后填充
        inter_df['借阅时间'] = inter_df.groupby('user_id')['借阅时间'].fillna(method='ffill').fillna(method='bfill')

    # 3. 填充还书时间：默认借阅后30天还书
    avg_lend_days = 30
    inter_df['还书时间'] = inter_df.apply(
        lambda row: row['借阅时间'] + pd.Timedelta(days=avg_lend_days)
        if pd.isnull(row['还书时间']) else row['还书时间'],
        axis=1
    )

    # 4. 续借逻辑：次数为0则续借时间空，次数>0则按比例生成续借时间
    inter_df['续借时间'] = inter_df.apply(
        lambda row:
        pd.NaT if row['续借次数'] == 0 else  # 续借次数为0 → 续借时间空
        (row['借阅时间'] + (row['还书时间'] - row['借阅时间']) * row['续借次数'] / (row['续借次数'] + 1))
        if pd.isnull(row['续借时间']) else row['续借时间'],  # 次数>0且时间空 → 按比例分配
        axis=1
    )

    # 5. 限制续借次数（超过5次设为5）
    inter_df['续借次数'] = inter_df['续借次数'].apply(lambda x: 5 if x > 5 else x)

    # 6. ID转为字符串（避免关联时类型不匹配）
    inter_df['user_id'] = inter_df['user_id'].astype(str)
    inter_df['book_id'] = inter_df['book_id'].astype(str)

    return inter_df


def clean_book_secondary_category(file_path):
    """
    清洗图书数据：填充缺失的二级分类（利用“相邻行二级分类相同”的排序特性，前向填充）
    参数：
        file_path: 图书数据文件路径（如CSV/Excel）
    返回：
        清洗后的DataFrame
    """
    # 读取数据（若为Excel，可替换为 pd.read_excel）
    book_df = pd.read_csv(file_path,encoding='utf-8')

    # 对“二级分类”列进行前向填充：用前一个非缺失值填充当前缺失值
    book_df['二级分类'] = book_df['二级分类'].fillna(method='ffill')

    return book_df

def main():
    # 测试借阅数据清洗函数
    # 1. 定义文件路径
    input_file = "../data/inter_preliminary.csv"   # 原始数据文件路径（需与脚本同目录，或写绝对路径）
    output_file = "test1.csv"  # 清洗后的数据输出路径

    # 2. 读取原始数据
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取 {input_file}，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取文件失败：{e}")
        return

    # 3. 调用清洗函数
    cleaned_df = clean_inter_data(input_file)

    # 4. 输出清洗后的数据到CSV
    cleaned_df.to_csv(output_file, index=False)  # index=False 表示不保存DataFrame的索引列
    print(f"数据清洗完成，已保存至 {output_file}，共 {len(cleaned_df)} 条记录")

    # 测试图书数据清洗
    input_file = "../data/item.csv"  # 替换为实际图书数据文件路径
    cleaned_book = clean_book_secondary_category(input_file)

    # 保存清洗后的数据（可选）
    cleaned_book.to_csv("test2.csv", index=False)
    print("二级分类缺失值已通过前向填充补全～")
if __name__ == "__main__":
    main()