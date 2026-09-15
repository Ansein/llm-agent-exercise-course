# 第01课完整教学样板

业务问题建模与传统NLP基线。输入客服问句，比较规则与字符TF-IDF＋逻辑回归的意图路由；涵盖数据隔离、未知拒识、成本、混淆矩阵与困难样本。

本包已提供真实可运行实验与保存输出。无需API、无需GPU，依赖安装后可离线运行。数据全部为合成教学材料，不能当作真实客服分布或生产上线证据。

## 先打开什么

| 使用者 | 入口 | 用途 |
|---|---|---|
| 讲师 | [教师讲义](教师讲义.md) | 180分钟脚本、原理推导、演示与常见追问 |
| 投影授课 | [20页PowerPoint](slides/第01课-业务问题建模与传统NLP基线.pptx) / [Markdown逐页稿](slides/逐页稿.md) | 可编辑课件、讲师备注和实测对照图 |
| 学员 | [实验手册](学员实验手册.md) | A—E实验、检查点与交付要求 |
| 交互实验 | [Notebook源文件](notebooks/01-intent-baselines.ipynb) | 按步骤执行、讨论与扩展 |
| 无Jupyter前端 | [已执行Notebook](notebooks/01-intent-baselines.executed.ipynb) / [静态HTML](notebooks/01-intent-baselines.html) | 查看已保存代码、表格和图表 |
| 结果核对 | [实测报告](reference/baseline-v1/baseline-report.md) | 指标、环境、20个错误表达族和24条挑战分析 |
| 评分 | [参考答案与评分](参考答案与评分.md) / [作业模板](作业报告模板.md) | 标准解读、100分量表与提交结构 |
| 数据审查 | [数据卡](data/DATA_CARD.md) / [manifest](data/manifest.json) | 字段、来源、标签边界、冻结hash与限制 |
| 复现审查 | [技术验收记录](reference/验收记录.md) | 独立环境复跑、测试与未验证事项 |

## 安装

从仓库根目录执行，推荐Python 3.12。以下为macOS/Linux shell示例，Windows需使用对应虚拟环境Python路径。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r courseware/lesson-01/requirements.txt
```

`requirements.txt`固定直接依赖。`requirements-lock.txt`为本次macOS arm64环境完整依赖快照，同平台复现可改用它；其他操作系统的传递依赖和二进制兼容性尚未验证，不能称为通用跨平台锁。

实测：Python 3.12.13，scikit-learn 1.7.2、NumPy 2.3.5、SciPy 1.16.3、pandas 2.2.3、Matplotlib 3.10.8。精确系统与硬件信息见[原始结果](reference/baseline-v1/results.json)。本地验证用临时隔离虚拟环境，没有安装到系统Python。

纯离线课堂需讲师提前在同平台下载wheel包，学员再从本地安装：

```bash
python3 -m pip download -r courseware/lesson-01/requirements-lock.txt -d wheelhouse
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r courseware/lesson-01/requirements-lock.txt
```

第一条需要网络，课前执行；wheelhouse未放入本仓库。不要在课堂才发现依赖或平台不匹配。

## 运行两条基线

```bash
.venv/bin/python courseware/lesson-01/src/evaluate.py --output courseware/lesson-01/runs/my-first-run
```

命令会核对数据hash→审计分组→只用训练集fit→验证集选阈值→测试集评测→生成报告、图表与逐样本输出。已有非空输出目录会拒绝覆盖；下一次换一个名字，不覆盖reference。

主要结果：`results.json`保存环境/配置/数据hash/时延原始值；`predictions.csv`包含所有280条主测试预测；`threshold-selection.csv`只含验证结果；`per-class-*.csv`给出逐类指标；`confusion-*.csv/png`保存矩阵；`challenge-results.csv`含24条挑战；`baseline-report.md`串起全部证据。

## 运行Notebook

```bash
.venv/bin/python courseware/lesson-01/src/execute_notebook.py --output-dir courseware/lesson-01/runs/notebook-run
```

脚本以当前Python创建临时内核，从上到下运行24个单元（11个代码单元），保存`.executed.ipynb`与自包含HTML，然后关闭内核。不需要安装JupyterLab。内核需要本机回环端口；如组织限制端口，在获准的Jupyter环境运行，CLI基线本身不依赖端口。

已安装Notebook前端的用户也可打开源文件并选择本虚拟环境内核。源Notebook无TODO代码洞，所有练习要求都在Markdown中；先跑通讲解版，再在自己的副本上实验。

## 检查与数据再生成

```bash
.venv/bin/python -m unittest discover -s courseware/lesson-01/tests -v
.venv/bin/python courseware/lesson-01/src/generate_data.py --output-dir courseware/lesson-01/runs/regenerated-data
```

12项检查覆盖客户/模板泄漏、归一化重复、冻结文件篡改、生成复现、训练词表隔离、拒识指标分母、验证成本选择、规则冲突、独立指标重算及输出防覆盖。生成器拒绝覆盖已有samples.csv，重新生成的hash应与manifest一致。

## 本次结果与解释

| 方案 | 已知macro-F1 | 已知覆盖率 | 未知误收 |
|---|---:|---:|---:|
| 规则 | 0.3385 | 37.50% | 0/40 |
| TF-IDF＋拒识 | 0.3661 | 40.42% | 0/40 |
| TF-IDF强制分类（闭集对照） | 0.6100 | 100% | 40/40 |

已知分母240；拒识阈值0.25来自验证集。阈值路线略高的F1没有带来更低教学成本，差值bootstrap区间跨0。测试刻意采用困难表达，低分用于发现失败，不通过改测试数据制造漂亮成绩。更多边界见报告与数据卡。

## 材料与维护

`src/`含生成、模型、评测、Notebook与课件构建源；数据和参考输出可直接带到下一课。课件可在PowerPoint中编辑，原生比较图附可编辑数据；重新生成PPTX需要Codex捆绑的Artifact Tool，Python实验环境不承担此依赖。完整逐页稿可独立维护，不影响分类实验复现。

讲义与课件中的数值以baseline-v1为准；更改数据、规则、模型或拒识策略，应生成新版本和新输出目录并更新材料，不静默替换历史。`src/build_notebook.py`可重建源Notebook，执行器只写指定输出目录；正式维护之前保留学员改动。

本样板完成技术交付与脚本复现；真实学员试讲、跨操作系统验证、生产运行仍未进行。Notebook整页HTML浏览器视觉验收的限制见验收记录。

[课程总览](../../README.md) · [本课大纲](../../lessons/01-业务问题建模与传统NLP基线.md)
