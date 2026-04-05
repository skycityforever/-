```
# 基于高校图书馆借阅数据的用户潜在图书推荐项目
本项目为“全球校园人工智能算法精英大赛”赛题六的实现代码，旨在通过机器学习/深度学习算法，预测用户未来潜在借阅需求并生成个性化推荐。

## 一、项目结构
详细结构见`library_book_recommendation/`目录，核心模块包括数据预处理、模型训练、评估、推荐结果生成。

## 二、环境搭建
1. 安装Python3.6+（推荐Python3.8）
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
```

```
library_book_recommendation/
├── data/                     # 数据存储目录
├── data_preprocessing/       # 数据预处理模块
├── model/                    # 模型定义与训练模块
├── evaluation/               # 模型评估模块
├── recommendation/           # 推荐结果生成模块
├── utils/                    # 工具函数模块
├── config/                   # 配置文件目录
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖库清单
└── README.md                 # 项目说明文档
```

```
data/
├── raw_data/                 # 原始数据（报名后下载的正式数据）
│   ├── book.csv              # 图书信息（含book_id、题名、作者等字段）{insert\_element\_0\_}
│   ├── inter.csv             # 借阅交互记录（含user_id、book_id、借阅时间等）{insert\_element\_1\_}
│   └── user.csv              # 用户信息（含性别、DEPT、年级等字段）{insert\_element\_2\_}
├── processed_data/           # 预处理后的数据
│   ├── train_data.csv        # 训练集数据
│   ├── valid_data.csv        # 验证集数据
│   └── test_data.csv         # 测试集数据（赛题提供的约14510条数据）{insert\_element\_3\_}
└── submission/               # 推荐结果输出
    └── submission.csv        # 最终提交文件（含user_id、book_id字段）{insert\_element\_4\_}
```

```data_preprocessing/
data_preprocessing/
├── __init__.py
├── data_cleaning.py          # 数据清洗（处理缺失值、异常值）
├── feature_engineering.py    # 特征工程（提取时间、分类等特征）
└── data_splitting.py         # 数据集划分（按赛题分阶段数据需求拆分）
```

```
model/
├── __init__.py
├── base_models/              # 基础模型（协同过滤、矩阵分解）
│   ├── user_cf.py            # 基于用户的协同过滤
│   └── wals.py               # 加权交替最小二乘法（WALS）
├── advanced_models/          # 进阶模型（GNN、序列推荐）
│   ├── lightgcn.py           # LightGCN模型（图协同过滤）
│   └── lstm_seq_rec.py       # LSTM序列推荐模型
└── model_trainer.py          # 模型训练与保存（含交叉验证、参数调优）
```

```evaluation/
evaluation/
├── __init__.py
├── metrics.py                # 评估指标计算（P、R、F1）
└── evaluate_model.py         # 模型评估主函数
```

```
recommendation/
├── __init__.py
└── generate_submission.py    # 生成提交文件
```

```
utils/
├── __init__.py
├── logger.py                 # 日志记录（训练过程、错误信息）
├── config_loader.py          # 配置文件加载（YAML格式）
└── visualization.py          # 可视化工具（特征重要性、推荐理由展示）
```

```
config/
└── config.yaml               # 配置文件
```
