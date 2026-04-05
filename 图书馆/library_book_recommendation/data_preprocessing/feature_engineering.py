import os

import pandas as pd
from datetime import datetime

def extract_time_features(inter_df):
    """提取时间特征：借阅月份、是否学期内（适配场景化兴趣）{insert\_element\_9\_}"""
    inter_df['借阅月份'] = inter_df['借阅时间'].dt.month
    # 定义学期范围（9-1月、2-7月为学期内，8月为假期）
    inter_df['是否学期内'] = inter_df['借阅月份'].apply(
        lambda x: 1 if (x in [9,10,11,12,1] or x in [2,3,4,5,6,7]) else 0
    )
    return inter_df

def build_interaction_weight(inter_df):
    """构建交互权重：续借次数越高，权重越大（体现兴趣强度）{insert\_element\_10\_}"""
    inter_df['交互权重'] = 1.0 + inter_df['续借次数'] * 0.2  # 续借1次加0.2权重
    return inter_df


def encode_categorical_features(user_df, book_df):
    """编码分类特征：用户院系、图书分类（适配模型输入）"""
    # 修复2：先将用户表“借阅人”重命名为“user_id”
    user_df = user_df.rename(columns={"借阅人": "user_id"})
    # 标签编码院系
    user_df['DEPT_encoded'] = user_df['DEPT'].astype('category').cat.codes
    # 图书分类编码
    book_df = pd.get_dummies(book_df, columns=['一级分类'], prefix='book_cat1')
    book_df['二级分类_encoded'] = book_df['二级分类'].astype('category').cat.codes
    return user_df, book_df


def compute_dept_book_preference(inter_df, user_df):
    """计算每个院系对图书的偏好度（借阅频次占比）"""
    # 修复3：用户表先重命名“借阅人”为“user_id”
    user_df = user_df.rename(columns={"借阅人": "user_id"})

    # 关联交互数据与用户院系
    inter_with_dept = pd.merge(
        inter_df[['user_id', 'book_id']],
        user_df[['user_id', 'DEPT']],
        on='user_id',
        how='left'
    )

    # 计算偏好度
    book_total = inter_with_dept.groupby('book_id').size().reset_index(name='total_borrow')
    dept_book_count = inter_with_dept.groupby(['DEPT', 'book_id']).size().reset_index(name='dept_borrow')
    dept_book_pref = pd.merge(dept_book_count, book_total, on='book_id', how='left')
    dept_book_pref['dept_book_preference'] = dept_book_pref['dept_borrow'] / (dept_book_pref['total_borrow'] + 1e-8)

    return dept_book_pref[['DEPT', 'book_id', 'dept_book_preference']]


def compute_dept_cat_preference(inter_df, user_df, book_df):
    """计算每个院系对图书分类（一级/二级）的偏好度"""
    # 修复4：用户表重命名“借阅人”为“user_id”
    user_df = user_df.rename(columns={"借阅人": "user_id"})

    # 关联交互数据+用户DEPT+图书分类
    inter_dept_cat = pd.merge(
        inter_df[['user_id', 'book_id']],
        user_df[['user_id', 'DEPT']],
        on='user_id',
        how='left'
    )
    inter_dept_cat = pd.merge(
        inter_dept_cat,
        book_df[['book_id', '一级分类', '二级分类']],
        on='book_id',
        how='left'
    )

    # 计算院系对一级分类偏好
    dept_total = inter_dept_cat.groupby('DEPT').size().reset_index(name='dept_total')
    dept_cat1_count = inter_dept_cat.groupby(['DEPT', '一级分类']).size().reset_index(name='dept_cat1_borrow')
    dept_cat1_pref = pd.merge(dept_cat1_count, dept_total, on='DEPT', how='left')
    dept_cat1_pref['dept_cat1_preference'] = dept_cat1_pref['dept_cat1_borrow'] / (dept_cat1_pref['dept_total'] + 1e-8)

    # 计算院系对二级分类偏好
    dept_cat2_count = inter_dept_cat.groupby(['DEPT', '二级分类']).size().reset_index(name='dept_cat2_borrow')
    dept_cat2_pref = pd.merge(dept_cat2_count, dept_total, on='DEPT', how='left')
    dept_cat2_pref['dept_cat2_preference'] = dept_cat2_pref['dept_cat2_borrow'] / (dept_cat2_pref['dept_total'] + 1e-8)

    return dept_cat1_pref, dept_cat2_pref


def add_user_dept_features(user_df, inter_df, book_df):
    """为用户表添加院系相关特征"""
    # 修复5：用户表重命名“借阅人”为“user_id”
    user_df = user_df.rename(columns={"借阅人": "user_id"})

    # 1. 计算院系平均续借次数
    user_inter = pd.merge(
        user_df[['user_id', 'DEPT']],
        inter_df[['user_id', '续借次数']],
        on='user_id',
        how='left'
    )
    dept_avg_renew = user_inter.groupby('DEPT')['续借次数'].mean().reset_index(name='dept_avg_renew')
    user_df = pd.merge(user_df, dept_avg_renew, on='DEPT', how='left')

    # 2. 计算院系对用户借阅分类的偏好
    user_book_cat = pd.merge(
        inter_df[['user_id', 'book_id']].drop_duplicates(),
        book_df[['book_id', '一级分类']],
        on='book_id',
        how='left'
    )
    user_book_cat = pd.merge(
        user_book_cat,
        user_df[['user_id', 'DEPT']],
        on='user_id',
        how='left'
    )
    # 调用分类偏好函数（已处理用户表列名）
    dept_cat1_pref, _ = compute_dept_cat_preference(inter_df, user_df.rename(columns={"user_id": "借阅人"}), book_df)
    user_dept_cat1_pref = pd.merge(
        user_book_cat,
        dept_cat1_pref[['DEPT', '一级分类', 'dept_cat1_preference']],
        on=['DEPT', '一级分类'],
        how='left'
    )
    user_avg_dept_cat1 = user_dept_cat1_pref.groupby('user_id')['dept_cat1_preference'].mean().reset_index(
        name='user_dept_avg_cat1_pref')
    user_df = pd.merge(user_df, user_avg_dept_cat1, on='user_id', how='left')

    return user_df


def merge_dept_features_to_inter(inter_df, user_df, book_df):
    """将院系相关特征合并到交互表中"""
    # ---------------------- 关键修改：删除交互表中已有的DEPT列（若存在） ----------------------
    inter_df = inter_df.drop(columns=['DEPT'], errors='ignore')  # errors='ignore'：列不存在时不报错
    # 修复6：用户表重命名“借阅人”为“user_id”
    user_df = user_df.rename(columns={"借阅人": "user_id"})

    # 1. 关联用户DEPT到交互表
    inter_with_dept = pd.merge(
        inter_df,
        user_df[['user_id', 'DEPT']],
        on='user_id',
        how='left'
    )

    # 2. 合并院系-图书偏好
    # 传入用户表时需还原为“借阅人”列（匹配compute_dept_book_preference的处理）
    dept_book_pref = compute_dept_book_preference(inter_df, user_df.rename(columns={"user_id": "借阅人"}))
    inter_with_dept = pd.merge(
        inter_with_dept,
        dept_book_pref,
        on=['DEPT', 'book_id'],
        how='left'
    )
    inter_with_dept['dept_book_preference'] = inter_with_dept['dept_book_preference'].fillna(0)

    # 3. 合并院系-分类偏好
    dept_cat1_pref, _ = compute_dept_cat_preference(inter_df, user_df.rename(columns={"user_id": "借阅人"}), book_df)
    inter_with_dept = pd.merge(
        inter_with_dept,
        book_df[['book_id']],
        on='book_id',
        how='left'
    )
    inter_with_dept = pd.merge(
        inter_with_dept,
        dept_cat1_pref[['DEPT', '一级分类', 'dept_cat1_preference']],
        on=['DEPT', '一级分类'],
        how='left'
    )
    inter_with_dept['dept_cat1_preference'] = inter_with_dept['dept_cat1_preference'].fillna(0)

    # 移除临时列
    inter_with_dept = inter_with_dept.drop(columns=['一级分类'])

    return inter_with_dept


def main():
    """测试特征工程函数：读取数据→执行所有特征工程→输出结果验证"""
    # 配置文件路径（根据实际情况修改）
    data_paths = {
        "inter": "../data/merged_data.csv",  # 交互数据（可替换为train_data.csv）
        "user": "../data/user.csv",  # 用户数据（列名含“借阅人”）
        "book": "../data/item.csv"  # 图书数据
    }
    output_path = "../data/feature_engineering_result/"
    encoding = "utf-8"  # 中文文件可改为'gbk'

    # 读取数据
    try:
        inter_df = pd.read_csv(
            data_paths["inter"],
            encoding=encoding,
            parse_dates=["借阅时间", "还书时间", "续借时间"]
        )
        user_df = pd.read_csv(data_paths["user"], encoding=encoding)
        book_df = pd.read_csv(data_paths["book"], encoding=encoding)

        print(f"数据读取成功：")
        print(f"交互数据：{len(inter_df)}行，列名：{inter_df.columns.tolist()[:5]}...")
        print(f"用户数据：{len(user_df)}行，列名：{user_df.columns.tolist()}")  # 确认“借阅人”列存在
        print(f"图书数据：{len(book_df)}行，列名：{book_df.columns.tolist()[:5]}...")
    except Exception as e:
        print(f"数据读取失败：{str(e)}")
        return

    # 执行特征工程
    try:
        # 时间特征
        inter_df = extract_time_features(inter_df)
        print("\n1. 时间特征提取完成")

        # 交互权重
        inter_df = build_interaction_weight(inter_df)
        print("2. 交互权重构建完成")

        # 分类特征编码
        user_df_encoded, book_df_encoded = encode_categorical_features(user_df.copy(), book_df.copy())
        print("3. 分类特征编码完成")

        # 院系-图书偏好
        dept_book_pref = compute_dept_book_preference(inter_df, user_df.copy())
        print(f"4. 院系-图书偏好计算完成（{len(dept_book_pref)}条）")

        # 院系-分类偏好
        dept_cat1_pref, dept_cat2_pref = compute_dept_cat_preference(inter_df, user_df.copy(), book_df.copy())
        print(f"5. 院系-分类偏好计算完成（一级{len(dept_cat1_pref)}条，二级{len(dept_cat2_pref)}条）")

        # 用户表增强
        user_df_enhanced = add_user_dept_features(user_df_encoded, inter_df, book_df.copy())
        print("6. 用户表特征增强完成")

        # 交互表合并特征
        inter_df_enhanced = merge_dept_features_to_inter(inter_df.copy(), user_df.copy(), book_df.copy())
        print("7. 交互表特征合并完成")

    except Exception as e:
        print(f"特征工程执行失败：{str(e)}")
        return

    # 结果验证
    print("\n" + "=" * 50)
    print("特征工程结果抽样：")

    # 交互表结果（含新特征）
    print("\n【增强后交互表（前3行）】")
    # dept_book_preference 字段的含义是：用户所在院系对当前图书的偏好度
    print(inter_df_enhanced[
              ['user_id', 'book_id', '借阅月份', '是否学期内', '交互权重', 'dept_book_preference']
          ].head(3).to_string(index=False))

    # 用户表结果（含新特征）
    print("\n【增强后用户表（前3行）】")
    # DEPT_encoded：对 “院系（DEPT）” 的标签编码
    # dept_avg_renew：用户所在院系的平均续借次数
    # user_dept_avg_cat1_pref：用户所在院系对该用户借阅过的 “一级图书分类”的平均偏好度
    print(user_df_enhanced[
              ['user_id', 'DEPT', 'DEPT_encoded', 'dept_avg_renew', 'user_dept_avg_cat1_pref']
          ].head(3).to_string(index=False))

    # 保存结果
    try:
        os.makedirs(output_path, exist_ok=True)
        inter_df_enhanced.to_csv(f"{output_path}inter_enhanced.csv", index=False, encoding=encoding)
        user_df_enhanced.to_csv(f"{output_path}user_enhanced.csv", index=False, encoding=encoding)
        book_df_encoded.to_csv(f"{output_path}book_encoded.csv", index=False, encoding=encoding)
        print(f"\n结果已保存至：{output_path}")
    except Exception as e:
        print(f"结果保存失败：{str(e)}")

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()