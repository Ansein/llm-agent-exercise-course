"""Reproduce the frozen lesson-01 reference. Run from any working directory."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import unicodedata
from datetime import datetime, timezone

# Bounded local caches; no changes to user-wide matplotlib/Jupyter settings.
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp'))/'agentcourse-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits
from baselines import (KeywordRules, make_model, scored_predict, apply_threshold,
                       choose_threshold, routing_cost, UNKNOWN, normalize)

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {
 'shipping_status':'物流查询','cancel_order':'取消订单','refund_status':'退款进度',
 'return_request':'退货申请','exchange_request':'换货申请','invoice':'发票问题',
 'payment':'支付问题','change_address':'修改收件信息','warranty':'保修维修',
 'technical_help':'安装使用','complaint':'服务投诉','product_info':'售前咨询', 'unknown':'拒识/未知'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def content_key(text):
    return ''.join(c for c in normalize(text) if not unicodedata.category(c).startswith('P'))


def validate_frame(df):
    required = {'sample_id','customer_id','template_id','text','intent','split','source','dataset_version'}
    if not required <= set(df.columns):
        raise ValueError(f'Missing fields: {required-set(df.columns)}')
    if df[list(required)].isna().any().any() or df.text.str.strip().eq('').any():
        raise ValueError('Null or empty input')
    if not df.sample_id.is_unique:
        raise ValueError('Duplicate sample ID')
    if not set(df.split) <= {'train','validation','test'}:
        raise ValueError('Invalid split')
    if not set(df.intent) <= set(LABEL_NAMES):
        raise ValueError('Invalid intent')
    for key in ['customer_id', 'template_id']:
        if (df.groupby(key).split.nunique() > 1).any():
            raise ValueError(f'Cross-split {key} leakage')
    if df.text.map(content_key).duplicated().any():
        raise ValueError('Duplicate normalized content')
    if ((df.split == 'train') & (df.intent == UNKNOWN)).any():
        raise ValueError('Unknown samples must not enter closed-set training')
    return {'rows':len(df), 'known_rows':int((df.intent != UNKNOWN).sum()),
        'split_counts':{k:int(v) for k,v in df.split.value_counts().items()},
        'customer_groups':int(df.customer_id.nunique()), 'template_groups':int(df.template_id.nunique()),
        'cross_split_customer_overlap':0, 'cross_split_template_overlap':0,
        'normalized_duplicates':0}


def load_data(data_dir=ROOT/'data'):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir/'manifest.json').read_text())
    if sha256(data_dir/'samples.csv') != manifest['sha256']:
        raise ValueError('Frozen dataset hash mismatch; create a new dataset version instead of overwriting')
    df = pd.read_csv(data_dir/'samples.csv', keep_default_na=False)
    audit = validate_frame(df)
    expected = {'train':720, 'validation':280, 'test':280}
    if audit['split_counts'] != expected or audit['known_rows'] != 1200:
        raise ValueError('Reference fixture counts mismatch')
    if not (df.dataset_version == manifest['dataset_version']).all():
        raise ValueError('Dataset version mismatch')
    for name, digest in manifest['source_hashes'].items():
        if sha256(data_dir/name) != digest:
            raise ValueError(f'Authored source changed without dataset version: {name}')
    return df, audit


def summary_metrics(truth, predictions, labels, costs):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    known = truth != UNKNOWN
    accepted = predictions != UNKNOWN
    return {
        'known_n':int(known.sum()), 'unknown_n':int((~known).sum()),
        'known_macro_f1':float(f1_score(truth[known], predictions[known], labels=labels, average='macro', zero_division=0)),
        'known_accuracy':float(accuracy_score(truth[known], predictions[known])),
        'known_coverage':float(accepted[known].mean()),
        'known_accepted_accuracy':float((truth[known & accepted] == predictions[known & accepted]).mean()) if (known & accepted).any() else None,
        'unknown_false_accept_rate':float(accepted[~known].mean()),
        'unknown_accepted_n':int(accepted[~known].sum()),
        'known_rejected_n':int((known & ~accepted).sum()),
        'routing_cost_per_request':float(routing_cost(truth, predictions, costs))}


def paired_cluster_bootstrap(frame, labels, iterations, seed):
    """Stratified family bootstrap; five variants never count as independent families."""
    rng = np.random.default_rng(seed)
    truth = frame.intent.to_numpy()
    predictions = [frame.rule_prediction.to_numpy(), frame.tfidf_prediction.to_numpy()]
    families = []
    for label in labels:
        family_list = [np.asarray(indices) for indices in frame[frame.intent == label].groupby('template_id').indices.values()]
        # groupby.indices above is local to a filtered frame; use global positional indices instead.
        family_list = [np.flatnonzero((frame.intent.to_numpy() == label) & (frame.template_id.to_numpy() == family))
                       for family in frame.loc[frame.intent == label, 'template_id'].unique()]
        families.append(family_list)
    values = []
    for _ in range(iterations):
        indices = np.concatenate([family_list[j] for family_list in families
                                   for j in rng.integers(0, len(family_list), size=len(family_list))])
        scores = [f1_score(truth[indices], pred[indices], labels=labels, average='macro', zero_division=0)
                  for pred in predictions]
        values.append([scores[0], scores[1], scores[1]-scores[0]])
    return {name: [float(v) for v in np.percentile(np.asarray(values)[:,i], [2.5,97.5])]
            for i,name in enumerate(['rules','tfidf','paired_difference'])}


def latency(predict_fn, texts):
    for text in texts[:10]:
        predict_fn([text])
    durations = []
    for text in texts:
        start=time.perf_counter_ns(); predict_fn([text]); durations.append((time.perf_counter_ns()-start)/1e6)
    return {'requests':len(texts), 'concurrency':1, 'batch_size':1, 'warmup':10,
            'p50_ms':float(np.percentile(durations,50)), 'p95_ms':float(np.percentile(durations,95)),
            'scope':'warm local classification call; no network, queue, business tools, or human processing',
            'raw_ms':durations}


def plot_figures(out, predictions, summaries, labels, trials, threshold):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.8), layout='constrained')
    keys=['known_macro_f1','known_coverage','unknown_false_accept_rate']
    names=['Known macro-F1','Known coverage','OOD false accept']
    xs=np.arange(3)
    for i,(model,color) in enumerate([('rules','#315C9B'),('tfidf','#C47B27')]):
        vals=[summaries[model][k] for k in keys]
        bars=ax.bar(xs+(i-.5)*.34, vals, .34, label=model, color=color)
        ax.bar_label(bars,labels=[f'{v:.1%}' for v in vals], padding=3)
    ax.set(xticks=xs,xticklabels=names,ylim=(0,1.15),ylabel='Rate (0–1)',title='Held-out routing results | 240 known + 40 OOD samples')
    ax.legend(loc='upper right');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    all_labels=labels+[UNKNOWN]
    for model in ['rules','tfidf']:
        cm=confusion_matrix(predictions.intent,predictions[f'{model[:-1] if model == "rules" else model}_prediction'],labels=all_labels)
        pd.DataFrame(cm,index=all_labels,columns=all_labels).to_csv(out/f'confusion-{model}.csv',encoding='utf-8')
        fig,ax=plt.subplots(figsize=(11,9),layout='constrained')
        display_labels=[f'{i+1:02}' for i in range(12)]+['OOD']
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=40)
        ax.set(xticks=np.arange(13),yticks=np.arange(13),xticklabels=display_labels,yticklabels=display_labels,
               xlabel='Predicted intent (OOD = abstain)',ylabel='True intent',title=f'{model}: confusion counts | n=280')
        for (r,c),v in np.ndenumerate(cm):
            ax.text(c,r,str(v),ha='center',va='center',color='white' if v>20 else '#263238',fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.75,label='Sample count')
        fig.savefig(out/f'confusion-{model}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(trials.threshold,trials.cost,'o-',color='#315C9B',label='Validation cost')
    ax.axvline(threshold,color='#8C5721',linestyle='--',label=f'Selected threshold = {threshold:.2f}')
    ax.set(xlabel='Max-score rejection threshold',ylabel='Illustrative cost per request',
           title='Threshold selection | validation only (240 known + 40 OOD)')
    ax.legend();fig.savefig(out/'threshold.png',dpi=150);plt.close(fig)


def run(output_dir, data_dir=ROOT/'data'):
    out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty. Choose a new directory to preserve previous evidence.')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'config.json').read_text())
    df,audit=load_data(data_dir)
    train=df[df.split=='train']; validation=df[df.split=='validation']; test=df[df.split=='test'].copy().reset_index(drop=True)
    labels=list(json.loads((ROOT/'data/templates.json').read_text()))
    rules=KeywordRules(); model=make_model(config)
    with threadpool_limits(limits=1):
        start=time.perf_counter();model.fit(train.text,train.intent);fit_seconds=time.perf_counter()-start
        val_labels,val_scores=scored_predict(model,validation.text)
        threshold,trials=choose_threshold(validation.intent,val_labels,val_scores,config)
        raw,scores=scored_predict(model,test.text)
        test['rule_prediction']=rules.predict(test.text)
        test['tfidf_raw_prediction']=raw;test['tfidf_score']=scores
        test['tfidf_prediction']=apply_threshold(raw,scores,threshold)
        test['rule_correct']=test.rule_prediction.eq(test.intent)
        test['tfidf_correct']=test.tfidf_prediction.eq(test.intent)
        summaries={key:summary_metrics(test.intent,test[col],labels,config['costs'])
                   for key,col in [('rules','rule_prediction'),('tfidf','tfidf_prediction'),('tfidf_forced','tfidf_raw_prediction')]}
        ci=paired_cluster_bootstrap(test[test.intent!=UNKNOWN].reset_index(drop=True),labels,config['bootstrap_iterations'],config['seed'])
        timings={'rules':latency(rules.predict,test.text.tolist()),
                 'tfidf':latency(lambda texts: apply_threshold(*scored_predict(model,texts),threshold),test.text.tolist())}
        # Diagnostic only: use the already train-fitted vectorizer, never refit on test.
        vec=model.named_steps['tfidf']; known=test[test.intent!=UNKNOWN]
        similarities=cosine_similarity(vec.transform(known.text),vec.transform(train.text))
        maxima=similarities.max(axis=1); nearest=similarities.argmax(axis=1)
    audit['near_duplicate_diagnostic']={'threshold':config['near_duplicate_cosine_threshold'],
        'test_known_rows':len(known),'flagged_rows':int((maxima>=config['near_duplicate_cosine_threshold']).sum()),
        'max_cosine':float(maxima.max()), 'median_nearest_cosine':float(np.median(maxima)),
        'method':'train-fitted char 2–4 TF-IDF cosine; flags are similarity diagnostics, not proof of semantic independence'}
    pd.DataFrame({'test_id':known.sample_id.to_numpy(),'train_id':train.sample_id.to_numpy()[nearest],
                  'cosine':maxima}).sort_values('cosine',ascending=False).to_csv(out/'nearest-train.csv',index=False)
    test.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(trials).to_csv(out/'threshold-selection.csv',index=False)
    pd.DataFrame(summaries).T.to_csv(out/'metrics.csv')
    per_class={}
    for name,col in [('rules','rule_prediction'),('tfidf','tfidf_prediction')]:
        report=classification_report(test.intent,test[col],labels=labels+[UNKNOWN],output_dict=True,zero_division=0)
        per_class[name]=report
        pd.DataFrame([{**{'intent':label,'name':LABEL_NAMES[label]},**report[label]} for label in labels+[UNKNOWN]]).to_csv(out/f'per-class-{name}.csv',index=False)
    challenges=pd.read_csv(ROOT/'data/challenges.csv',keep_default_na=False)
    challenges['rule_prediction']=rules.predict(challenges.text)
    c_raw,c_scores=scored_predict(model,challenges.text)
    challenges['tfidf_prediction']=apply_threshold(c_raw,c_scores,threshold)
    challenges['tfidf_score']=c_scores
    challenges.to_csv(out/'challenge-results.csv',index=False)
    environment={'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
        'logical_cpu_count':os.cpu_count(),'thread_limit':1,
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','pandas','matplotlib']}}
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,'selected_threshold':threshold,
        'audit':audit,'metrics':summaries,'bootstrap_95_percent':ci,'latency':timings,'fit_seconds':fit_seconds,
        'vocabulary_size':len(model.named_steps['tfidf'].vocabulary_),'environment':environment,
        'label_order':labels,'dataset_sha256':sha256(Path(data_dir)/'samples.csv'),
        'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'config.json',ROOT/'data/rules.json',ROOT/'data/challenges.csv',ROOT/'src/baselines.py',ROOT/'src/evaluate.py']}}
    save_json(out/'results.json',result)
    save_json(out/'data-audit.json',audit)
    save_json(out/'per-class.json',per_class)
    plot_figures(out,test,summaries,labels,pd.DataFrame(trials),threshold)
    generate_report(out,result,test,challenges)
    return result


def generate_report(out,r,test,challenges):
    rows=[]
    for name in ['rules','tfidf','tfidf_forced']:
        m=r['metrics'][name]
        rows.append(f"| {name} | {m['known_macro_f1']:.4f} | {m['known_accuracy']:.1%} | {m['known_coverage']:.1%} | {m['unknown_false_accept_rate']:.1%} ({m['unknown_accepted_n']}/40) | {m['routing_cost_per_request']:.4f} |")
    errors=test[(test.intent != test.rule_prediction)|(test.intent != test.tfidf_prediction)].drop_duplicates('template_id').head(20)
    error_rows='\n'.join(f'| {x.sample_id} | {x.text} | {x.intent} | {x.rule_prediction} | {x.tfidf_prediction} |' for x in errors.itertuples())
    legend='；'.join(f'{i+1:02}={LABEL_NAMES[label]}' for i,label in enumerate(r['label_order']))
    crows='\n'.join(f'| {x.case_id} | {x.text} | {x.expected_intent} | {x.rule_prediction} | {x.tfidf_prediction} | {x.reason} |' for x in challenges.itertuples())
    text=f'''# 第01课基线实测报告

## 数据与方法

数据版本1.0.0；SHA256：`{r['dataset_sha256']}`。1200条已知意图来自240个手写核心表达，每个扩写5种礼貌包装；另有80条未知意图。训练720条，验证240已知+40未知，测试240已知+40未知。测试的48个已知表达族是有效分组，240行不是240个独立语言样本。

规则由关键词唯一命中决定；零命中或多意图命中拒识。TF-IDF使用字符2—4 gram，逻辑回归C=4，训练词表只见训练数据。验证集按错路由3、已知拒识1、未知误收5的教学成本选阈值，锁定为 **{r['selected_threshold']:.2f}**；这些成本是教学假设，不是企业核定金额。测试结果未用于修改规则、特征、C或阈值。

## 冻结测试结果

| 方案 | 已知macro-F1 | 已知准确率 | 已知覆盖率 | 未知误收率 | 每请求教学成本 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

已知类指标分母240，未知误收分母40。拒识的已知问题按错误计入F1；unknown代表停止自动路由并交人工，不能把拒识说成自动解决。`tfidf_forced`不拒识，只作闭集对照；未知请求被强行分到12类是其固有限制。

![方案比较](comparison.png)

## 不确定性与阈值

按意图分层、按表达族配对bootstrap {r['config']['bootstrap_iterations']}次，95%百分位区间：规则F1 {r['bootstrap_95_percent']['rules']}；带拒识TF-IDF F1 {r['bootstrap_95_percent']['tfidf']}；两者差值 {r['bootstrap_95_percent']['paired_difference']}。区间只描述该合成表达池的抽样变化，不包含真实业务分布偏移或多次重新训练的不确定性。

![验证集阈值](threshold.png)

## 数据隔离与剩余相似性

客户跨集合数0，表达族跨集合数0，去标点/空白归一后完全重复数0。训练拟合的TF-IDF近重复诊断：相似度≥0.85有{r['audit']['near_duplicate_diagnostic']['flagged_rows']}/240条，最高{r['audit']['near_duplicate_diagnostic']['max_cosine']:.3f}；见[最近训练样本](nearest-train.csv)。客户ID与表达族在本合成数据中一一对应，真实业务需独立核查两种分组。通用包装、关键词、语义仍可能相似，分组无交叉不等于没有分布偏差。

## 运行环境与耗时

Python {r['environment']['python']}，{r['environment']['platform']}；{r['environment']['machine']}，逻辑CPU {r['environment']['logical_cpu_count']}，数值计算线程限制1。版本见[原始结果](results.json)。训练耗时{r['fit_seconds']:.4f}秒，词表{r['vocabulary_size']}项。

串行单条请求，预热10条后测280条：规则P50/P95={r['latency']['rules']['p50_ms']:.4f}/{r['latency']['rules']['p95_ms']:.4f}ms；TF-IDF={r['latency']['tfidf']['p50_ms']:.4f}/{r['latency']['tfidf']['p95_ms']:.4f}ms。这是本地分类调用时间，包含文本处理、预测及拒识，不含网络、排队、业务执行或人工时间；不是服务SLA。无模型API调用费，不等于总业务成本为零。

## 混淆矩阵与错误样本

数字图例：{legend}；OOD=unknown。行是真实标签，列是预测标签，矩阵包含全部280条。

![规则混淆矩阵](confusion-rules.png)

![TF-IDF混淆矩阵](confusion-tfidf.png)

下面按文件顺序列出最多20个不同表达族的失败（任一方案失败即入选），完整结果见[predictions.csv](predictions.csv)。它们是诊断样例，不是额外调参集。

| ID | 输入 | 真实 | 规则 | TF-IDF |
|---|---|---|---|---|
{error_rows}

## 24条人工设计困难案例

独立教学挑战集，不参与训练、阈值选择或上述280条主指标。expected_intent为本课程标注规范下的目标：多诉求/无上下文指代设为unknown，表示先澄清。每条的成因说明在运行前编写，预测列为实际运行所得。

| ID | 输入 | 预期 | 规则 | TF-IDF | 边界及改进方向 |
|---|---|---|---|---|---|
{crows}

## 结论与边界

应根据质量、覆盖率与误收共同选择方案，而非仅比较F1。规则对冲突采取保守拒识，TF-IDF可能吸收更多表达但仍无法可靠理解否定、多个诉求和上下文。该数据中的未知意图主要是明显域外问题，低误收不代表能识别贴近业务边界的所有未知请求；挑战集用于暴露这一限制。

本实验测量的是意图路由，不能据此宣称客服自动解决率、满意度或处理时长已经改善。下一步可在新开发集研究否定、槽位与小模型，对原冻结测试集仅做回归，不以反复调试后的成绩冒充首次泛化。实测数值和时延可由CLI复跑；浮点尾数和时延允许受硬件影响。
'''
    (out/'baseline-report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    r=run(args.output,args.data_dir)
    print(json.dumps({'selected_threshold':r['selected_threshold'],'metrics':r['metrics'],'output':str(args.output)},ensure_ascii=False,indent=2))
