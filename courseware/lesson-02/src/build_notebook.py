"""构建第02课可执行Notebook（只生成源文件，不执行）。"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md('''# 第02课实验：张量、文本编码与最小自动微分

## Goal · 学习目标

把一段客服文本变成张量，手写一次前向与反向，训练一个embedding均值池化分类器，
并与第01课的TF-IDF基线在同一冻结测试集上对照。重点不是把分数做高，而是：

1. 说清 `batch × sequence × hidden` 三个维度分别是什么；
2. 用数值梯度证明自己写的反向是对的；
3. 用mask证明补齐不改变有效文本的表示；
4. 能定位"补齐参与均值"和"标签错位"这两类看似正常的错误。

**已执行参考结论**：强制分类的已知macro-F1为0.7456（第01课TF-IDF为0.6100）；
带拒识后为0.4926、覆盖率52.9%，与第01课的配对差值区间跨0。
梯度校验最大相对误差5.2e-05，masked mean在不同补齐宽度下池化漂移为0。
本Notebook无待补代码，练习题在Markdown中；不需要GPU或API。''')

md('''## Setup · 环境与假设

按教学包README在Python 3.12虚拟环境安装依赖。本Notebook可从仓库根目录、教学包或notebooks目录启动。

### 关键假设

- 数据沿用第01课冻结的客服意图数据（1280条、客户与表达族隔离），本课**不重新生成**数据，
  否则无法与第01课基线公平对照。
- 词表只从训练集构建：`<pad>`索引0、`<unk>`索引1；验证与测试的未见字符记UNK。
- 训练集不含未知意图样本，闭集训练与拒识评测分开。''')

code('''from pathlib import Path
import json, sys, tempfile
import numpy as np
import pandas as pd
from IPython.display import display, Image

cwd = Path.cwd().resolve()
search = [cwd, *cwd.parents]
ROOT = next((p for p in search if (p / "src/autograd.py").is_file()), None)
if ROOT is None:
    ROOT = next(p / "courseware/lesson-02" for p in search
                if (p / "courseware/lesson-02/src/autograd.py").is_file())
sys.path.insert(0, str(ROOT / "src"))
from autograd import Tensor, backward, embedding, linear, masked_mean, numerical_gradients, relative_error, softmax_cross_entropy
from data import LABEL_NAMES, UNKNOWN, load_frozen, split_frames, load_challenges
from encoding import PAD_ID, UNK_ID, CharVocab, encode_corpus, pad_batch, tokenize
from evaluate import (UNKNOWN_ID, encode_labels, gradient_check, padding_sensitivity,
                      batch_width_invariance, choose_threshold, run)
from model import MeanPoolClassifier, ModelConfig, one_hot, train_model, macro_f1
config = json.loads((ROOT / "config.json").read_text())
print("Python", sys.version.split()[0], "| 训练配置", config["training"])''')

md('''## Steps · 实验步骤

### 1. 读取冻结数据并核对分母

`load_frozen`会校验第01课冻结文件的hash、数量、客户与表达族隔离。校验失败先修数据来源，
不要跳过检查；也不要为了让分数变好而改动测试集。''')

code('''frame, manifest, labels, audit = load_frozen()
print("数据版本", manifest["dataset_version"], "| sha256", manifest["sha256"][:16], "...")
display(pd.Series({k: v for k, v in audit.items() if k != "split_counts"}, name="audit"))
display(pd.crosstab(frame["intent"], frame["split"]).reindex(list(LABEL_NAMES)[:-1] + ["unknown"]))
train, validation, test = split_frames(frame)
assert len(train) == 720 and audit["known_test_rows"] == 240 and audit["unknown_test_rows"] == 40''')

md('''### 2. 文本 → ID → 张量：三个维度

字符级编码的好处是没有分词依赖，代价是序列变长。补齐把同一batch的序列对齐成矩阵，
mask同时记录哪些位置是真实token。**这三样东西的维度必须能对上**：
`ids/mask`是 `(batch, sequence)`，查表后是 `(batch, sequence, hidden)`，
池化后回到 `(batch, hidden)`，分类头输出 `(batch, classes)`。''')

code('''tokenizer = CharVocab(min_count=config["training"]["min_count"],
                       ngram_max=config["training"]["ngram_max"]).fit(train.text)
demo = train.text.head(4).tolist()
demo_ids, demo_mask = pad_batch([tokenizer.encode(t) for t in demo])
print("ids 形状", demo_ids.shape, "| mask 形状", demo_mask.shape)
for text, ids, mask in zip(demo, demo_ids, demo_mask):
    print(f"  {text}  ->  {len(tokenizer.encode(text))} tokens")
display(pd.DataFrame({"token": [tokenizer.id_to_token[i] for i in demo_ids[0][:12]],
                      "id": demo_ids[0][:12], "mask": demo_mask[0][:12]}))
print("PAD_ID =", PAD_ID, "| UNK_ID =", UNK_ID, "| 词表大小", tokenizer.vocab_size)''')

md('''**观察题**：为什么不能把 `template_id` 也当作特征喂给模型？（提示：编号里带着标签，属于泄漏。）
为什么词表必须只从训练集构建？''')

md('''### 3. 补齐与attention mask

mask为1的位置参与平均，为0的位置被排除。下面用同一个embedding表检查：
同一个句子被补到更宽的画布后，池化向量是否改变。''')

code('''rng = np.random.default_rng(7)
table = Tensor(rng.normal(0, 0.1, (tokenizer.vocab_size, 32)), requires_grad=True)
ids, mask = pad_batch([tokenizer.encode(t) for t in demo])
embedded = embedding(table, ids)
pooled = masked_mean(embedded, mask)
print("embedding查表后", embedded.shape, "-> 池化后", pooled.shape)

# 只扩宽、不截断：正确实现下池化向量不应变化。
wide_ids, wide_mask = pad_batch([tokenizer.encode(t) for t in demo], max_length=ids.shape[1] + 40)
wide_pooled = masked_mean(embedding(table, wide_ids), wide_mask)
print("补齐到", wide_ids.shape[1], "后池化最大差异:",
      float(np.abs(pooled.data - wide_pooled.data).max()))''')

md('''### 4. 全流程形状与参数量

按 `(batch, sequence, hidden)` 逐层打印形状，并计算与第01课稀疏词袋对应的参数量。''')

code('''train_arrays = encode_corpus(tokenizer, train.text, config["max_length"])[:2]
validation_arrays = encode_corpus(tokenizer, validation.text, config["max_length"])[:2]
test_arrays = encode_corpus(tokenizer, test.text, config["max_length"])[:2]
label_index = {label: i for i, label in enumerate(labels)}
train_labels = encode_labels(train.intent, label_index)
validation_labels = encode_labels(validation.intent, label_index)
test_labels = encode_labels(test.intent, label_index)

shapes = MeanPoolClassifier(tokenizer.vocab_size, len(labels), ModelConfig(**config["training"]))
print("参数量", shapes.parameter_count())
logits, pooled_vectors, effective = shapes.forward(train_arrays[0][:8], train_arrays[1][:8])
print("(batch, seq)", train_arrays[0][:8].shape,
      "-> (batch, seq, hidden)", embedding(shapes.embedding_table, train_arrays[0][:8]).shape,
      "-> (batch, hidden)", pooled_vectors.shape,
      "-> (batch, classes)", logits.shape)
assert pooled_vectors.shape == (8, config["training"]["hidden_dim"])
assert logits.shape == (8, len(labels))''')

md('''### 5. 手写反向：数值梯度校验

反向传播必须从损失出发、沿计算图回传。顺序写错时中间梯度会保持0，训练照样"跑完"但学不到东西。
所以每次实现新的算子，都要用中心差分数值梯度对一次。

softmax交叉熵对logits的梯度是 `(p − y) / N`；线性层是 `dW = xᵀ·dY`、`dx = dY·Wᵀ`；
masked mean按有效长度归一，所以回传时要除以有效长度；embedding查表的梯度用 `np.add.at` 累加到对应行。''')

code('''check_model = MeanPoolClassifier(tokenizer.vocab_size, len(labels), ModelConfig(**config["training"]))
report = gradient_check(check_model, train_arrays[0], train_arrays[1], train_labels,
                        n_rows=config["gradient_check_rows"])
display(pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "note"}
                      for k, v in report.items() if isinstance(v, dict)}).T)
print("最大相对误差:", max(v["max_relative_error"] for v in report.values() if isinstance(v, dict)))''')

md('''**读表**：偏置的误差最小（1e-9量级），因为它不经过embedding求和；embedding与分类权重在
1e-5量级，这正是中心差分本身在 `epsilon=1e-6`、float64下的精度水平。
若某个参数的相对误差接近1，说明它的解析梯度是错的（例如反向顺序反了或某条路径没回传）。''')

md('''### 6. 训练最小分类器

Adam + 分批训练。每个epoch在验证集（仅已知类）上记录loss与macro-F1，
用于观察"训练很好但验证不升"这一常见现象。''')

code('''model = MeanPoolClassifier(tokenizer.vocab_size, len(labels), ModelConfig(**config["training"]))
known_validation = validation_labels != UNKNOWN_ID
history = train_model(model, train_arrays, train_labels,
                      (validation_arrays[0][known_validation], validation_arrays[1][known_validation]),
                      validation_labels[known_validation], verbose=True)
display(pd.DataFrame(history.as_dict()).tail(5).round(4))''')

md('''### 7. 完整评测并与第01课对照

`run`是CLI的底层函数：它重新按同一配置训练、在验证集上选拒识阈值、在冻结测试集上评测，
并导出图表与报告。Notebook里把它写到临时目录，避免覆盖冻结参考。

口径与第01课完全一致：已知类macro-F1分母240条，拒识的已知样本按错误计入；
未知误收分母40条；强制分类作为闭集对照。''')

code('''output = Path(tempfile.mkdtemp(prefix="lesson02-notebook-"))
result = run(output)
metrics = pd.DataFrame(result["metrics"]).T
display(metrics[["known_macro_f1", "known_accuracy", "known_coverage",
                 "unknown_false_accept_rate", "routing_cost_per_request"]].round(4))
print("验证集选出的拒识阈值:", result["selected_threshold"])
display(Image(filename=str(output / "comparison.png"), width=850))''')

md('''**读图**：均值池化把强制分类的macro-F1从0.6100提升到0.7456，说明"可训练的稠密表示"
比稀疏词袋更能吸收同义表达；但带拒识后F1只有0.4926、覆盖率52.9%，
而且它把4/40条域外请求误收进了业务类别（第01课TF-IDF为0/40）。
这不是"更好或更差"，而是**质量、覆盖率与误收三者的取舍**。''')

code('''display(pd.DataFrame(result["bootstrap_95_percent"], index=["2.5%", "97.5%"]).T.round(4))
display(Image(filename=str(output / "training-curves.png"), width=900))
display(Image(filename=str(output / "confusion-mini.png"), width=850))''')

md('''### 8. 注入缺陷：补齐参与均值

把补齐位置也算进平均，是初学者最常见的实现错误。它不会让程序报错，甚至可能让指标变好，
但会让同一句话的表示依赖batch内最长样本。下面的检查用「只扩宽、不截断」把这种不稳定暴露出来。''')

code('''sensitivity = padding_sensitivity(model, test_arrays[0], test_arrays[1], test_labels, config["max_length"])
invariance = batch_width_invariance(model, test_arrays[0], test_arrays[1], test_labels)
print("同一批样本，正确实现 vs 缺陷实现：")
print("  池化最大差异:", round(sensitivity["max_pooled_abs_difference"], 4),
      "| 预测改变条数:", sensitivity["prediction_changes"], "/ 280")
display(pd.DataFrame(invariance).T[["widths", "max_pooled_drift", "prediction_changes_across_widths"]])''')

md('''**结论**：正确实现（masked mean）在不同补齐宽度下池化漂移为0；
把补齐计入均值后漂移达0.34，并且同一句话的预测会随batch组成变化。
真实系统里这表现为"离线评测正常、线上单条推理却出错"。''')

md('''### 9. 注入缺陷：标签错位

把标签整体错位一位，输入与标签不再对应。它和"标签读错列"、“shuffle后忘记同步标签”是同一类错误。
诊断信号是：训练loss几乎不下降（接近 `ln(类别数)`），预测集中在少数类别。''')

code('''from evaluate import run_experiment
shifted = run_experiment(train_arrays, train_labels, validation_arrays, validation_labels,
                         test_arrays, test_labels, tokenizer.vocab_size, len(labels),
                         ModelConfig(**config["training"]), label_shift=1)
aligned_forced = (result["metrics"]["mini_mean_pool_forced"]["known_macro_f1"])
shifted_forced = macro_f1(test_labels[test_labels != UNKNOWN_ID],
                          shifted["test_predictions"][test_labels != UNKNOWN_ID], len(labels))
display(pd.DataFrame({
    "配置": ["正确对齐", "标签错位1位"],
    "末轮训练loss": [history.train_loss[-1], shifted["history"].train_loss[-1]],
    "验证loss": [history.validation_loss[-1], shifted["history"].validation_loss[-1]],
    "强制分类macro-F1": [aligned_forced, shifted_forced]}).round(4))''')

md('''### 10. 失败样本与困难案例

均值池化丢掉了词序，因此对否定、状态条件和多诉求最弱。下面看两类样本：
主测试集里的失败表达族，以及24条独立的困难挑战集。''')

code('''predictions = pd.read_csv(output / "predictions.csv", keep_default_na=False)
errors = predictions[predictions.intent != predictions.mini_prediction].drop_duplicates("template_id")
print("失败的表达族数:", len(errors))
display(errors[["sample_id", "text", "intent", "mini_prediction", "length"]].head(8))
challenges = pd.read_csv(output / "challenge-results.csv", keep_default_na=False)
display(challenges[["case_id", "text", "expected_intent", "mini_prediction", "reason"]].iloc[[0, 3, 10, 14, 15, 19]])''')

md('''**分析**：`H15 写一首包含退款和发票的诗`、`H16 帮我翻译这句话：我要取消订单` 这类样本
含业务词但并非业务请求；词袋与均值池化都靠词形，无法区分"引用"和"执行"。
`H20 已经收到了，这一单不要了` 需要状态条件，词袋模型看不到"已签收"与"未发货"的区别。''')

md('''## Checks · 复现与口径检查

与冻结参考比较：数据hash、编码配置、阈值与质量指标。时延与浮点尾数允许受硬件影响，
所以不比较逐位相等。测试集已公开用于教学，后续BERT/GPT若反复接触它，泛化结论需另建未见集合。''')

code('''reference = json.loads((ROOT / "reference/mini-autograd-v1/results.json").read_text())
assert result["dataset_sha256"] == reference["dataset_sha256"]
assert result["selected_threshold"] == reference["selected_threshold"]
assert result["encoding"]["vocab_size"] == reference["encoding"]["vocab_size"]
for name, values in result["metrics"].items():
    for metric, value in values.items():
        expected = reference["metrics"][name][metric]
        if value is not None:
            assert np.isclose(value, expected, atol=1e-10), (name, metric)
print("参考口径已复现：数据hash、阈值、词表与全部质量指标一致。无API调用、无生产写入。")''')

md('''## Next Steps · 练习与交付

1. 推导并手算一个两样本batch的 `dW` 与 `db`，与 `gradient_check` 的结果对照。
2. 把均值池化改成"取最大"或"加权池化"，说明它对长度和否定词的影响；**只用验证集选择**。
3. 把 `ngram_max` 改为2，记录词表大小、序列长度与验证指标的变化，写出一页资源取舍结论。
4. 复现"补齐参与均值"缺陷，解释为什么它的测试F1可能更高，以及上线后会在什么情况下出错。
5. 回答：为什么意图分类正确，仍然不能说明客服自动解决率提升了？

参考实现与实际结果属于教学样板。数据为合成教学材料，未做真实客户数据、独立讲师复现或生产上线。

资料：[Transformer原论文](https://arxiv.org/abs/1706.03762) · [技术参考与时效说明](../../docs/技术参考.md)''')

nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12'}}
nbf.validate(nb)
nbf.write(nb, ROOT / 'notebooks/02-tensors-and-text-representation.ipynb')
print(f'Notebook scaffold: {len(cells)} cells')
