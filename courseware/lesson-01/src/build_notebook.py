"""Build the portable teaching notebook using nbformat (does not execute it)."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# 第01课实验：客服意图分类的两条基线

## Goal · 学习目标

把业务问题变成意图路由实验，比较规则与字符TF-IDF＋逻辑回归；正确处理数据隔离、拒识和结果解释。

**已执行参考结论**：在该合成困难测试上，规则/带拒识TF-IDF的已知macro-F1为0.3385/0.3661，覆盖率37.5%/40.4%。这不支持“上线自动客服”；两者差值区间跨0。参考结果用于核对运行，不能拿它调测试集。

本Notebook是可运行讲解版，无待补代码；练习题在Markdown中，学员另写实验分支。无需API或GPU。''')
md('''## Setup · 环境与假设

先按教学包README在Python 3.12虚拟环境安装依赖。本Notebook可从仓库根目录、教学包或notebooks目录启动。仅在本地临时目录生成此次评测结果，不覆盖冻结参考包。

### 关键假设

1200条已知意图由240个核心表达各扩写5条；未知意图80条另计。客户与表达族一一对应，仅用于演示分组隔离。测试最后4个表达族/类刻意包含否定、同义与背景干扰，属于困难表达迁移实验，不是随机生产抽样。规则和模型只吃text，所有ID仅用于审计。''')
code('''from pathlib import Path
import sys, json, tempfile
import numpy as np
import pandas as pd
from IPython.display import display, Image

cwd = Path.cwd().resolve()
search = [cwd, *cwd.parents]
ROOT = next((p for p in search if (p / "src/baselines.py").is_file()), None)
if ROOT is None:
    ROOT = next(p / "courseware/lesson-01" for p in search
                if (p / "courseware/lesson-01/src/baselines.py").is_file())
sys.path.insert(0, str(ROOT / "src"))
from baselines import KeywordRules, make_model, scored_predict, apply_threshold, choose_threshold
from evaluate import load_data, run, LABEL_NAMES
from threadpoolctl import threadpool_limits
config = json.loads((ROOT / "config.json").read_text())
print("Python", sys.version.split()[0], "| dataset version", config["dataset_version"])
''')
md('''## Steps · 实验步骤

### 1. 读取冻结数据并核对分母

`load_data`检查文件hash、空值、归一化重复、客户跨集合、表达族跨集合和数量。若失败先修复数据来源，不跳过校验。''')
code('''data, audit = load_data()
display(pd.crosstab(data["intent"], data["split"]).reindex(LABEL_NAMES.keys()))
display(pd.Series({k:v for k,v in audit.items() if k != "split_counts"}, name="audit"))
train = data[data.split == "train"]
validation = data[data.split == "validation"]
test = data[data.split == "test"]
assert len(train) == 720 and len(test[test.intent != "unknown"]) == 240
assert set(train.customer_id).isdisjoint(test.customer_id)
assert set(train.template_id).isdisjoint(test.template_id)
''')
md('''**观察题**：测试中每个已知类20行，但只有4个核心表达族。若把5种礼貌包装看成5个独立语言现象，会如何影响置信区间？未知意图40条不应进入已知12类macro-F1的分母。

### 2. 检查规则为什么拒识

唯一意图命中才路由；无命中或多个意图命中交人工。该机制不理解否定，因此拒识并不意味着真的识别出用户的主诉求。''')
code('''rules = KeywordRules()
examples = ["开具发票", "不要取消订单，只是改收货地址", "写一首包含退款和发票的诗"]
display(pd.DataFrame({"text":examples,
                     "matched": [rules.explain(t) for t in examples],
                     "prediction":rules.predict(examples)}))
''')
md('''### 3. 只在训练集拟合TF-IDF和逻辑回归

字符2—4 gram省去中文分词依赖，但不自动理解词序、否定和状态。训练流程为：文本→字符片段计数→IDF与L2归一化→线性分类分数。验证与测试只调用transform/predict。

**课堂练习**：用3条短句手算一个片段的平滑IDF：`log((1+n)/(1+df))+1`。完整推导见教师讲义。''')
code('''model = make_model(config)
with threadpool_limits(limits=1):
    model.fit(train.text, train.intent)
vectorizer = model.named_steps["tfidf"]
print("Training rows:", len(train), "| vocabulary:", len(vectorizer.vocabulary_))
terms = vectorizer.get_feature_names_out()
display(pd.DataFrame({"term":terms[:12], "idf":vectorizer.idf_[:12]}))
assert "customer_id" not in terms
''')
md('''### 4. 在验证集选择拒识阈值

未校准的最大类别分数低于阈值则交人工。教学成本：已知错路由3、已知拒识1、未知误收5。只对预先给出的阈值网格选最小验证成本，平局依次选择更少未知误收、更高已知覆盖、较低阈值。

这些权重和验证集未知占比影响选择；它们不是企业真实成本。''')
code('''val_labels, val_scores = scored_predict(model, validation.text)
threshold, trials = choose_threshold(validation.intent, val_labels, val_scores, config)
trial_table = pd.DataFrame(trials)
print("Selected validation threshold:", threshold)
display(trial_table.head(8).round(4))
''')
md('''### 5. 一次完整评测并保存可追溯结果

调用同一CLI底层函数，重新拟合相同配置、计算280条测试结果、逐类指标、表达族bootstrap和本地时延，并导出图表。其训练与阈值必须和上方教学步骤一致。

临时结果用于本次Notebook；正式提交请用README中的CLI保存到自己的runs目录。''')
code('''output = Path(tempfile.mkdtemp(prefix="lesson01-notebook-"))
result = run(output)
assert result["selected_threshold"] == threshold
predictions = pd.read_csv(output / "predictions.csv")
raw, scores = scored_predict(model, test.text)
assert np.array_equal(apply_threshold(raw, scores, threshold), predictions.tfidf_prediction)
metrics = pd.DataFrame(result["metrics"]).T
columns = ["known_macro_f1", "known_accuracy", "known_coverage", "unknown_false_accept_rate", "routing_cost_per_request"]
display(metrics[columns].round(4))
''')
md('''### 6. 对照质量、覆盖率和域外误收

蓝色规则、橙色TF-IDF；已知指标n=240，域外误收n=40。0次域外误收只是这个明显域外样本集上的结果，下一节挑战集会展示带业务词的域外请求仍然可能被误收。''')
code('''display(Image(filename=str(output / "comparison.png"), width=850))
display(Image(filename=str(output / "threshold.png"), width=850))
''')
md('''**读图结论**：拒识减少未知误收，同时牺牲已知覆盖。这里TF-IDF F1略高，但预定义成本并未优于规则；不能将更高F1直接等同更好的业务方案。

### 7. 逐类召回与混淆

矩阵行是真实、列是预测；OOD列表示拒识。以未拒识的240条强制分类成绩替代含拒识成绩，会改变口径。图中数字类名可由下面的表查阅。''')
code('''display(pd.DataFrame({"code":[f"{i:02}" for i in range(1,13)]+["OOD"],
                      "intent":list(LABEL_NAMES), "中文":list(LABEL_NAMES.values())}))
display(Image(filename=str(output / "confusion-rules.png"), width=850))
display(Image(filename=str(output / "confusion-tfidf.png"), width=850))
display(pd.read_csv(output / "per-class-tfidf.csv").round(3))
''')
md('''### 8. 不把模板扩写当独立样本

每个意图内对4个测试表达族有放回抽样，再合并该族5条问句，配对重采样1000次。比较方案差值的区间，而非只看两个单独区间是否重叠。''')
code('''display(pd.DataFrame(result["bootstrap_95_percent"], index=["2.5%", "97.5%"]).T.round(4))
display(pd.Series(result["audit"]["near_duplicate_diagnostic"]))
''')
md('''**结论**：配对F1差值区间跨0，无法据此断言TF-IDF稳定优于规则。相似度阈值未发现近重复，也不能证明语义独立；共享包装和合成语料仍造成局限。

### 9. 困难案例：先看意图，再看预测

24条挑战样本已在评测前编写，包含否定、缩写、多诉求、上下文缺失和业务词诱导的域外问题。完整逐例分析见输出报告，下面只展示6条。''')
code('''challenges = pd.read_csv(output / "challenge-results.csv")
display(challenges[["case_id","text","expected_intent","rule_prediction","tfidf_prediction","reason"]].iloc[[0,2,10,14,15,23]])
errors = predictions[(predictions.intent != predictions.rule_prediction) |
                     (predictions.intent != predictions.tfidf_prediction)].drop_duplicates("template_id")
print("Different failed known/unknown expression families:", len(errors))
display(errors[["sample_id","text","intent","rule_prediction","tfidf_prediction"]].head(5))
''')
md('''## Checks · 复现与口径检查

与冻结参考比较数据hash、阈值与质量指标。时延不比较精确相等，因为硬件、调度和负载会变化。测试集已公开用于教学；后续BERT/GPT若反复接触它，最终泛化需另建未见集合。''')
code('''reference = json.loads((ROOT / "reference/baseline-v1/results.json").read_text())
assert result["dataset_sha256"] == reference["dataset_sha256"]
assert result["selected_threshold"] == reference["selected_threshold"]
for method in result["metrics"]:
    for metric, value in result["metrics"][method].items():
        expected = reference["metrics"][method][metric]
        if value is not None:
            assert np.isclose(value, expected, atol=1e-10), (method, metric)
print("Reference quality metrics reproduced. No API calls or production actions.")
''')
md('''## Next Steps · 练习与交付

1. 提交一页业务指标树：意图分流正确率、知识命中率和端到端自动解决率分别如何测量？
2. 从独立错误表达族中选至少20例，写出错误类型、主诉求、风险及下一步验证。不能用加上测试答案关键词宣称泛化提升。
3. 在开发副本中比较字符1—2 gram和2—4 gram，或修改教学错误成本；**只用验证集选择**。最终测试另行冻结、只运行一次。
4. 回答为什么两个方案都不适合直接自动办理退款，以及BERT或GPT能改善哪些问题、仍需哪些规则控制。

参考实现与实际结果属于教学样板。没有真实客户数据、独立人工试讲或生产上线证据。

资料：[scikit-learn文本特征](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)、[数据泄漏](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage)。''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nbf.validate(nb)
nbf.write(nb,ROOT/'notebooks/01-intent-baselines.ipynb')
print(f'Notebook scaffold: {len(cells)} cells')
