# ELCRec: End-to-End Learnable Clustering for Sequential Recommendation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架实现的 ELCRec 模型，针对现有意图学习方法中聚类与表征学习分离导致次优解及扩展性差的问题，通过端到端的可学习聚类模块（ELCM）和意图辅助对比学习（ICL）统一了用户行为编码与潜在意图挖掘。

## 🌟 核心特点

- **端到端可学习聚类模块（ELCM）**：通过可学习的聚类中心参数，实现聚类与表征学习的统一优化
- **意图辅助对比学习（ICL）**：结合意图信息的对比学习，提升序列表征质量
- **模块化设计**：基于 ReChorus 框架，易于扩展和实验
- **完整实验验证**：在 MovieLens-1M 和 Amazon Grocery 数据集上验证了模型有效性

## 📋 目录结构

```
ReChorus-master/
├── src/                          # 源代码目录
│   ├── main.py                  # 主程序入口
│   ├── models/                   # 模型实现
│   │   └── sequential/
│   │       ├── ELCRec.py        # ELCRec 完整模型
│   │       ├── ELCRec_ELCM.py   # 消融实验：仅ELCM
│   │       └── ELCRec_ICL.py    # 消融实验：仅ICL
│   ├── helpers/                  # 数据读取和训练辅助类
│   └── utils/                    # 工具函数
├── data/                         # 数据集目录
│   ├── MovieLens_1M/           # MovieLens-1M 数据集
│   └── Grocery_and_Gourmet_Food/ # Amazon Grocery 数据集
├── requirements.txt             # 依赖包列表
└── README.md                    # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.10.4
- PyTorch 1.12.1
- CUDA 10.2+ (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone <your-repo-url>
cd ReChorus-master
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据集**

数据集应放置在 `data/` 目录下。项目已包含预处理好的 MovieLens-1M 和 Grocery_and_Gourmet_Food 数据集。

### 运行模型

#### MovieLens-1M 数据集

```bash
python src/main.py --model_name ELCRec --dataset MovieLens_1M --emb_size 64 --lr 0.001 --cluster_k 256 --alpha 0.1 --gpu 0 --num_workers 0
```

#### Amazon Grocery 数据集

```bash
python src/main.py --model_name ELCRec --dataset Grocery_and_Gourmet_Food --emb_size 64 --lr 0.001 --cluster_k 256 --alpha 0.1 --gpu 0 --num_workers 0
```

### 主要参数说明

- `--model_name`: 模型名称，使用 `ELCRec`
- `--dataset`: 数据集名称，可选 `MovieLens_1M` 或 `Grocery_and_Gourmet_Food`
- `--emb_size`: 嵌入向量维度，默认 64
- `--lr`: 学习率，默认 0.001
- `--cluster_k`: 聚类中心数量，默认 256
- `--alpha`: 聚类损失权重，默认 0.1
- `--gpu`: GPU设备ID，使用 `0` 表示第一块GPU，`-1` 或 `''` 表示使用CPU
- `--num_workers`: 数据加载线程数，Windows系统建议设为 0

### 消融实验

项目还提供了两个消融实验模型：

1. **ELCRec_ELCM**：仅保留端到端可学习聚类模块
```bash
python src/main.py --model_name ELCRec_ELCM --dataset MovieLens_1M --emb_size 64 --lr 0.001 --cluster_k 256 --alpha 0.1 --gpu 0
```

2. **ELCRec_ICL**：仅保留意图辅助对比学习
```bash
python src/main.py --model_name ELCRec_ICL --dataset MovieLens_1M --emb_size 64 --lr 0.001 --cluster_k 256 --w_cl 0.1 --tau 0.2 --gpu 0
```

## 🔬 模型架构

ELCRec 模型主要由以下组件构成：

1. **序列编码器**：基于 Transformer 的序列编码，提取用户行为序列的表示
2. **端到端可学习聚类模块（ELCM）**：
   - 可学习的聚类中心参数
   - 聚类损失：包含解耦损失（decouple）和对齐损失（align）
3. **意图辅助对比学习（ICL）**：
   - 序列对比学习（SeqCL）：基于增强视图的对比学习
   - 意图对比学习（IntentCL）：结合意图信息的对比学习
   - 意图融合：通过 shift 或 concat 方式融合意图信息

### 损失函数

总损失由三部分组成：

```
L = L_next + α * L_cluster + w_cl * (L_seqcl + L_intent)
```

- `L_next`: 基础推荐损失（BPR损失）
- `L_cluster`: 聚类损失（ELCM）
- `L_seqcl`: 序列对比损失
- `L_intent`: 意图对比损失

## 📊 实验结果

在 MovieLens-1M 和 Amazon Grocery 数据集上的实验表明，ELCRec 模型在多数指标上优于基准模型（SASRec、GRU4Rec等）。详细的实验结果请参考项目日志文件（`log/` 目录）。

## 🛠️ 技术细节

### 数据增强策略

模型使用三种数据增强策略：
- **Mask**：随机掩码部分序列项
- **Crop**：随机裁剪连续子序列
- **Reorder**：随机打乱部分序列顺序

### 意图融合方式

支持两种意图融合方式：
- **Shift**：直接相加 `h + c`
- **Concat**：拼接后通过线性层投影

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{elcrec2024,
  title={ELCRec: End-to-End Learnable Clustering for Sequential Recommendation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/your-repo}}
}
```

本项目基于 ReChorus 2.0 框架开发，请同时引用：

```bibtex
@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](./LICENSE) 文件。

## 🙏 致谢

- 感谢 [ReChorus](https://github.com/THUwangcy/ReChorus) 团队提供的优秀框架
- 感谢所有贡献者和用户的支持

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至：[dusr@mail2.sysu.edu.cn]

---

**注意**：本项目为学术研究用途，请确保遵守相关数据集的使用协议。
