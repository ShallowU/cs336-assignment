# CS336: Language Models from Scratch 

> 斯坦福CS336 (Spring 2025)课程作业的大部分实现，涵盖从零构建语言模型、系统优化、到模型对齐的全流程。

[课程官网](https://stanford-cs336.github.io/spring2025/)           [Youtube视频](https://youtu.be/SQ3fZ1sAqXI?si=nEylRQnHJNjot_FA)          [课程讲义](https://github.com/stanford-cs336/spring2025-lectures)       

##  项目概览

本仓库包含三个核心作业的实现，由于本人是快速刷课和做project的邪修，基础只有深度学习基础和刷过Andrej Karpathy大神的[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)，所以基本上代码都是由AI和参考其他大佬的代码和文档，包括[Sherlock1956佬的代码](https://github.com/Sherlock1956)和[哎哟薇佬的文档](https://github.com/weiruihhh/cs336_note_and_hw)。核心是先快速过一遍每个assignment，理解代码意思然后大部分手抄，最后实践动手跑通，有点类似一周速刷Leetcode Hot100。实验环境为colab，大部分用A100 40/80G即可。

| 作业         | 主题          | 核心内容                                          |
| ------------ | ------------- | ------------------------------------------------- |
| Assignment 1 | **Basics**    | 从零实现 Transformer LM、BPE 分词器、训练流程     |
| Assignment 2 | **Systems**   | Flash Attention、分布式训练 (DDP/FSDP)、性能优化  |
| Assignment 5 | **Alignment** | SFT 微调、EI、GRPO 强化学习对齐、数学推理能力提升 |

## colab链接和博客

- Assignment 1：[作业一博客](https://zhoudianfu.github.io/blogs/cs336-assignment1/)     [colab code-BPE Tokenizer](https://colab.research.google.com/drive/1fwKl0zUZVh6K6nIW4BT6Pz0RZ_ruQa2V?usp=sharing)            [colab code-TinyStories Pre-train](https://colab.research.google.com/drive/1In23AQT6isNRsoW82i4K-M8eCo300aZg?usp=sharing)
- Assignment 2:  [作业二博客](https://zhoudianfu.github.io/blogs/cs336-assignment2/)       [colab code-FlashAttention2](https://colab.research.google.com/drive/1CQtwWbOW2jj33R5mIZasAB6jR0e96WAH?usp=sharing)          [kaggle code-DDP](https://www.kaggle.com/code/shallowu/ddp-benchmark)
- Assignment 5:  [作业五博客](https://zhoudianfu.github.io/blogs/cs336-assignment5/)       [colab code-SFT,EI,GRPO](https://colab.research.google.com/drive/1mqC8IhEb8J1e2TdakORk1XKwR01KW9aN?usp=sharing)

## 快速开始

环境配置非常简单，使用uv即可，比pip快很多，在colab上时间就是金钱。当我们需要做某个assignment时候，cd到对应文件夹下按照该文件夹下官方的README配置步骤即可。这里以作业一为例：

```shell
cd assignment1/ 
```

```shell
uv sync # 一步直接同步环境
```

这里跑一下所有测试（应该全部Failed）：

```shell
uv run pytest
```

下载实验数据：

```shell
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

##  项目结构

```
cs336-assignment/
├── assignment1/          # 基础：Transformer + BPE + 训练
│   ├── cs336_basics/     # 核心实现
│   │   ├── layer.py      # Transformer 各层实现
│   │   ├── bpe.py        # BPE 分词器训练
│   │   ├── tokenizer.py  # 分词器推理
│   │   ├── optimizer.py  # AdamW 优化器
│   │   ├── loss.py       # 交叉熵损失
│   │   ├── data.py       # 数据加载
│   │   └── util.py       # 检查点保存/加载
│   └── train/            # 训练与生成脚本
│
├── assignment2/          # 系统优化：注意力 + 分布式
│   ├── cs336_systems/    # 优化实现
│   │   ├── layer.py      # 优化后的 Transformer 层
│   │   ├── flash_forward.py  # Triton Flash Attention
│   │   ├── naive_ddp.py      # 朴素分布式数据并行
│   │   ├── ddp_overlap_*.py  # 通信-计算重叠 DDP
│   │   ├── fsdp2.py          # 全分片数据并行
│   │   ├── ShardedOptimizer.py # 分片优化器
│   │   └── benchmark.py      # 性能基准测试
│   └── cs336-basics/     # Assignment 1 参考实现
│
└── assignment5/          # 对齐：SFT + EI +GRPO
    ├── cs336_alignment/  # 对齐实现
    │   ├── utils.py          # 核心工具函数
    │   ├── utils_assignment.py # 作业接口实现
    │   ├── sft.py            # SFT 监督微调
    │   ├── grpo.py           # GRPO 强化学习训练
    │   ├── config.py         # GRPO 训练配置
    │   ├── math_baseline.py  # GSM8K 评估
    │   └── drgrpo_grader.py  # 数学答案评分器
    ├── data/             # 数据集 (GSM8K, MMLU 等)
    └── scripts/          # 评估脚本
```

## Honor Code

- **Collaboration**: Study groups are allowed, but students must understand and complete their own assignments, and hand in one assignment per student. If you worked in a group, please put the names of the members of your study group at the top of your assignment. Please ask if you have any questions about the collaboration policy.
- **AI tools**: Prompting LLMs such as ChatGPT is permitted for low-level programming questions or high-level conceptual questions about language models, but using it directly to solve the problem is prohibited. We strongly encourage you to disable AI autocomplete (e.g., Cursor Tab, GitHub CoPilot) in your IDE when completing assignments (though non-AI autocomplete, e.g., autocompleting function names is totally fine). We have found that AI autocomplete makes it much harder to engage deeply with the content.
- **Existing code**: Implementations for many of the things you will implement exist online. The handouts we'll give will be self-contained, so that you will not need to consult third-party code for producing your own implementation. Thus, you should not look at any existing code unless when otherwise specified in the handouts.



## LICENSE
Apache License 2.0
