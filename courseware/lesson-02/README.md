# 第02课完整教学包：张量计算与文本表示基础

从字符到张量：文本编码、补齐与attention mask、手写自动微分、embedding均值池化分类器，
并与第01课TF-IDF基线在同一冻结测试集上对照。含两个可复现的实现缺陷：补齐参与均值、标签错位。

本包已提供真实可运行实验与保存输出。**不需要PyTorch**，纯NumPy实现前向与反向；无需GPU与API，依赖安装后可离线运行。
数据沿用第01课冻结的合成客服问句，本课不重新生成数据，运行时会校验hash。

## 先打开什么

| 使用者 | 入口 | 用途 |
|---|---|---|
| 讲师 | [教师讲义](教师讲义.md) | 180分钟脚本、形状与梯度推导、演示与常见追问 |
| 投影授课 | [20页课件](slides/第02课-张量计算与文本表示基础.pptx) / [逐页稿](slides/逐页稿.md) | 可编辑课件与讲师备注 |
| 学员 | [实验手册](学员实验手册.md) | A—F实验、检查点与交付要求 |
| 交互实验 | [Notebook源文件](notebooks/02-tensors-and-text-representation.ipynb) | 按步骤执行、改算子、做参数实验 |
| 无Jupyter前端 | [已执行Notebook](notebooks/02-tensors-and-text-representation.executed.ipynb) / [静态HTML](notebooks/02-tensors-and-text-representation.html) | 查看已保存代码、输出与图表 |
| 结果核对 | [实测报告](reference/mini-autograd-v1/baseline-report.md) | 指标、梯度校验、缺陷对照与失败样本 |
| 评分 | [参考答案与评分](参考答案与评分.md) / [作业模板](作业报告模板.md) | 标准答案、100分量表与提交结构 |
| 复现审查 | [编码审计](reference/mini-autograd-v1/encoding-audit.json) / [梯度校验](reference/mini-autograd-v1/gradient-check.json) | 词表、长度、OOV与数值梯度证据 |

## 安装

两课依赖相同，若已按第01课安装可跳过。从仓库根目录执行，推荐Python 3.12。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r courseware/lesson-01/requirements.txt
```

第01课教学包必须存在且未被改动：本课直接读取`courseware/lesson-01/data/samples.csv`与
`courseware/lesson-01/reference/baseline-v1/predictions.csv`，并校验manifest中的sha256。
哈希不一致会直接报错，这是有意设计：改数据后必须新建数据集版本，而不是就地覆盖。

## 运行

### 1. 完整实验（训练、评测、缺陷复现、报告与图表）

```bash
.venv/bin/python courseware/lesson-02/src/evaluate.py --output courseware/lesson-02/runs/my-first-run
```

流程：校验数据hash → 构建字符词表 → 编码并对齐 → 数值梯度校验 → 训练 →
验证集选拒识阈值 → 冻结测试评测 → 缺陷复现 → 导出报告与图表。
已有非空输出目录会被拒绝覆盖；下一次换一个新目录名，不覆盖reference。

### 2. 检查

```bash
.venv/bin/python -m unittest discover -s courseware/lesson-02/tests -v
```

25项检查覆盖：梯度数值一致性、反向拓扑顺序、mask补齐不变性、词表越界与UNK、
标签缺失拒绝、标签错位诊断、同seed可复现、指标分母口径、缺陷可区分性、
冻结hash篡改拒绝、参考结果可重算与输出防覆盖。

### 3. Notebook

```bash
.venv/bin/python courseware/lesson-02/src/execute_notebook.py --output-dir courseware/lesson-02/runs/notebook-run
```

以当前Python创建临时内核执行32个单元（13个代码单元），保存`.executed.ipynb`与自包含HTML后关闭内核。
内核需要本机回环端口；如组织限制端口，可只用CLI实验，CLI不依赖端口。
重建源Notebook用`src/build_notebook.py`，重建课件用`src/build_slides.py`。

## 本次结果与解释

| 方案 | 已知macro-F1 | 已知覆盖率 | 未知误收 | 每请求教学成本 |
|---|---:|---:|---:|---:|
| 均值池化＋拒识（本课） | 0.4926 | 52.92% | 4/40 | 0.7857 |
| 均值池化强制分类（闭集对照） | 0.7456 | 100% | 40/40 | 1.3464 |
| 第01课TF-IDF＋拒识（冻结预测） | 0.3661 | 40.42% | 0/40 | 0.8107 |
| 第01课TF-IDF强制分类（闭集对照） | 0.6100 | 100% | 40/40 | 1.7107 |

已知类分母240、未知误收分母40；拒识的已知样本按错误计入F1；阈值0.95由验证集按教学成本选出。
判断是**质量、覆盖率与误收三者的取舍**：可训练稠密表示把强制分类F1从0.6100提升到0.7456，
但代价是它对域外请求更自信，误收4/40（第01课为0/40）。两者差值的配对bootstrap区间跨0，
因此本课不宣称均值池化稳定优于TF-IDF。这些是合成困难测试上的结果，不支持生产效果承诺。

## 两个缺陷复现（本课重点）

| 配置 | 强制分类macro-F1 | 末轮训练loss | 关键证据 |
|---|---:|---:|---|
| 正确实现（masked mean） | 0.7456 | 0.0001 | 只扩宽不截断时池化漂移0.0 |
| 补齐参与均值 | 0.7879 | 0.0001 | 池化最大差异0.5243，30/280预测改变；扩宽时漂移0.3401 |
| 标签错位1位 | 0.0915 | 1.6388 | 输入与标签不再对应，退化为随机 |

「补齐参与均值」的测试F1**反而更高**。这不是正确性证据：它的真实代价是同一句话的表示
随batch内最长样本漂移，离线评测正常而线上单条推理出错。这个转折是本课要建立的判断习惯。

## 材料与维护

`src/`含编码、自动微分、模型、评测、Notebook与课件构建源。`reference/mini-autograd-v1/`为本次冻结输出，
含逐样本预测、训练历史、两种缺陷诊断与全部图表；讲义与课件数值以该目录为准。
更改编码、模型、优化器或数据，应生成新的runs目录与新版本说明，不静默替换历史。

## 已验证与未验证边界

已验证：CPU离线完整流程、25项自动检查、Notebook全量执行、课件包结构与内嵌图表数据检查、
数据hash与参考指标可重算一致。

未验证：未做独立讲师复现与学员试讲；未在原生PowerPoint打开本课课件（课件由标准库OOXML生成，
仅完成包结构与XML解析检查，放映效果、字体替换需在授课环境核对）；
未验证Windows/Linux依赖、GPU/API路线或生产流程。数据为合成教学材料，不等于真实客服分布。

[课程总览](../../README.md) · [本课大纲](../../lessons/02-张量计算与文本表示基础.md) · [第01课教学包](../lesson-01/README.md)
