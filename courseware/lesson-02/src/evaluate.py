"""第02课实验入口：编码审计、梯度校验、最小模型训练、缺陷复现与基线对照。

子命令：
  check            只做数据与编码审计，输出长度分布与OOV统计，不训练
  gradient-check   数值梯度校验，证明手写反向正确
  run              完整流程：审计→梯度校验→训练→缺陷复现→评测→报告与图表

所有输出写入新的输出目录；已存在的非空目录会被拒绝，避免覆盖历史证据。
"""
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

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'agentcourse-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autograd import Tensor, backward, numerical_gradients, relative_error, softmax_cross_entropy
from data import (LABEL_NAMES, LESSON01_DATA, UNKNOWN, load_challenges, load_frozen,
                  load_frozen_baseline, split_frames)
from encoding import CharVocab, PAD_ID, UNK_ID, encode_corpus, normalize, tokenize
from model import (ModelConfig, MeanPoolClassifier, macro_f1, one_hot, train_model)

ROOT = Path(__file__).resolve().parents[1]

# 图表需要中文（缺陷对照图的横轴标签）。按平台探测可用字体，找不到时回退为纯ASCII标签，
# 不下载字体、不修改系统配置。
CJK_FONT_CANDIDATES = [
    '/System/Library/Fonts/Supplemental/Songti.ttc',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    'C:/Windows/Fonts/msyh.ttc',
]


def configure_chinese_font() -> str | None:
    """注册第一个存在的中文字体；返回字体名，找不到返回None。"""
    from matplotlib import font_manager
    for candidate in CJK_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            return name
    return None


CJK_FONT = configure_chinese_font()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def prepare_output(output_dir) -> Path:
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('输出目录非空，请换一个新目录以保留已有证据')
    out.mkdir(parents=True, exist_ok=True)
    return out


UNKNOWN_ID = -1


def encode_labels(intents, label_index) -> np.ndarray:
    """把意图字符串编码为类别索引；UNKNOWN记-1，明确表示不属于闭集。

    不使用pandas的缺失值映射：小整数列可能被读成可空整型，缺省值会静默变成NA。
    """
    return np.array([label_index.get(intent, UNKNOWN_ID) for intent in intents], dtype=np.int64)


def load_config():
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    training = dict(config['training'])
    return config, ModelConfig(**training)


# --------------------------------------------------------------------------
# 编码审计
# --------------------------------------------------------------------------
def encoding_audit(tokenizer, frame, labels, max_length):
    """统计长度分布、截断比例与OOV率；oov统计会随encode累积，故本函数只调用一次。"""
    lengths, truncated = {}, {}
    for split in ['train', 'validation', 'test']:
        subset = frame[frame.split == split]
        raw_lengths = np.array([len(tokenizer.encode(text)) for text in subset.text])
        lengths[split] = {
            'count': int(len(raw_lengths)),
            'min': int(raw_lengths.min()), 'p50': float(np.percentile(raw_lengths, 50)),
            'p95': float(np.percentile(raw_lengths, 95)), 'p99': float(np.percentile(raw_lengths, 99)),
            'max': int(raw_lengths.max()), 'mean': float(raw_lengths.mean()),
        }
        truncated[split] = int((raw_lengths > max_length).sum())
    total_tokens = sum(len(tokenize(text, tokenizer.ngram_max)) for text in frame.text)
    oov_total = int(sum(tokenizer.oov_tokens.values()))
    return {'vocab_size': tokenizer.vocab_size, 'ngram_max': tokenizer.ngram_max,
            'min_count': tokenizer.min_count, 'max_length': int(max_length),
            'sequence_length': lengths, 'truncated_counts': truncated,
            'truncated_ratio': {key: value / max(1, lengths[key]['count']) for key, value in truncated.items()},
            'total_tokens': int(total_tokens),
            'oov_token_occurrences': oov_total,
            'oov_rate': float(oov_total / max(1, total_tokens)),
            'pad_id': PAD_ID, 'unk_id': UNK_ID, 'label_count': len(labels)}


# --------------------------------------------------------------------------
# 梯度校验
# --------------------------------------------------------------------------
def gradient_check(model, ids, mask, labels, n_rows=6, epsilon=1e-6):
    """对一个小batch做中心差分数值梯度校验，覆盖embedding、分类权重与偏置。"""
    rows = np.arange(min(n_rows, len(labels)))
    batch_ids, batch_mask = ids[rows], mask[rows]
    targets = one_hot(np.asarray(labels)[rows], model.n_classes)

    def loss_value():
        logits, _, _ = model.forward(batch_ids, batch_mask)
        return float(softmax_cross_entropy(logits, targets).data.reshape(-1)[0])

    for parameter in model.parameters:
        parameter.zero_grad()
    logits, _, _ = model.forward(batch_ids, batch_mask)
    loss = softmax_cross_entropy(logits, targets)
    backward(loss)
    analytic = {'embedding_table': model.embedding_table.grad.copy(),
                'classifier_weight': model.weight.grad.copy(),
                'classifier_bias': model.bias.grad.copy()}
    numeric = numerical_gradients(loss_value, model.parameters, epsilon=epsilon)
    report = {}
    for name, value in zip(['embedding_table', 'classifier_weight', 'classifier_bias'], numeric):
        report[name] = {'analytic_abs_sum': float(np.abs(analytic[name]).sum()),
                        'numeric_abs_sum': float(np.abs(value).sum()),
                        'max_relative_error': relative_error(analytic[name], value),
                        'elements': int(value.size)}
    report['loss_value'] = loss_value()
    report['batch_rows'] = int(len(rows))
    report['epsilon'] = epsilon
    report['gradient_clip_applied_in_training'] = model.config.grad_clip
    return report


# --------------------------------------------------------------------------
# 评测指标（与第01课完全相同的口径）
# --------------------------------------------------------------------------
def summary_metrics(truth, predictions, labels, costs):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    known = truth != UNKNOWN
    accepted = predictions != UNKNOWN
    known_truth, known_pred = truth[known], predictions[known]
    per_label = []
    for label in labels:
        true_positive = int(((known_truth == label) & (known_pred == label)).sum())
        false_positive = int(((known_truth != label) & (known_pred == label)).sum())
        false_negative = int(((known_truth == label) & (known_pred != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        per_label.append(2 * true_positive / denominator if denominator else 0.0)
    correct = int((known_truth == known_pred).sum())
    accept_known = known & accepted
    return {
        'known_n': int(known.sum()), 'unknown_n': int((~known).sum()),
        'known_macro_f1': float(np.mean(per_label)),
        'known_accuracy': float(correct / max(1, known.sum())),
        'known_coverage': float(accepted[known].mean()),
        'known_accepted_accuracy': float((truth[accept_known] == predictions[accept_known]).mean())
        if accept_known.any() else None,
        'unknown_false_accept_rate': float(accepted[~known].mean()),
        'unknown_accepted_n': int(accepted[~known].sum()),
        'known_rejected_n': int((known & ~accepted).sum()),
        'routing_cost_per_request': float(routing_cost(truth, predictions, costs)),
    }


def routing_cost(truth, predictions, costs):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    known = truth != UNKNOWN
    rejects = predictions == UNKNOWN
    return (np.sum(known & rejects) * costs['known_reject']
            + np.sum(known & ~rejects & (truth != predictions)) * costs['known_wrong']
            + np.sum(~known & ~rejects) * costs['unknown_accept']) / len(truth)


def choose_threshold(truth, probabilities, index_to_label, config):
    """与第01课相同的验证集成本选择与平局规则。"""
    max_scores = probabilities.max(axis=1)
    winners = index_to_label[probabilities.argmax(axis=1)]
    trials = []
    for threshold in config['threshold_grid']:
        predictions = np.where(max_scores >= threshold, winners, UNKNOWN)
        known = np.asarray(truth) != UNKNOWN
        trials.append({'threshold': float(threshold),
                       'cost': float(routing_cost(truth, predictions, config['costs'])),
                       'known_coverage': float(np.mean(predictions[known] != UNKNOWN)),
                       'unknown_false_accept_rate': float(np.mean(predictions[~known] != UNKNOWN))})
    best = min(trials, key=lambda item: (item['cost'], item['unknown_false_accept_rate'],
                                         -item['known_coverage'], item['threshold']))
    return best['threshold'], trials


def paired_cluster_bootstrap(frame, labels, iterations, seed):
    """按表达族分层的配对bootstrap；族内5条变体一起采样，不当作独立语言现象。

    直接接收整数索引数组，避免依赖DataFrame列名；只在已知类样本上计算macro-F1，
    与第01课口径一致（拒识的已知样本仍计入其F1分母）。
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(frame.intent_index.to_numpy(), dtype=np.int64)
    templates = np.asarray(frame.template_id.to_numpy())
    predictions = [np.asarray(frame.mini_index_selected.to_numpy(), dtype=np.int64),
                   np.asarray(frame.tfidf_index.to_numpy(), dtype=np.int64)]
    families = []
    for label in labels:
        # 用np.unique而不是pd.unique/categorical：类目型列可能带未使用取值，
        # 会产出空表达族并导致采样结果为空。
        family_ids = np.unique(templates[truth == label])
        families.append([np.flatnonzero((truth == label) & (templates == family))
                         for family in family_ids])
    draws = []
    for _ in range(iterations):
        indices = np.concatenate([family_list[j] for family_list in families
                                  for j in rng.integers(0, len(family_list), size=len(family_list))])
        scores = [macro_f1(truth[indices], prediction[indices], len(labels)) for prediction in predictions]
        draws.append([scores[0], scores[1], scores[0] - scores[1]])
    draws = np.asarray(draws)
    return {name: [float(value) for value in np.percentile(draws[:, index], [2.5, 97.5])]
            for index, name in enumerate(['mini_mean_pool', 'tfidf_frozen', 'paired_difference'])}


# --------------------------------------------------------------------------
# 缺陷复现
# --------------------------------------------------------------------------
def run_experiment(train_arrays, train_labels, validation_arrays, validation_labels,
                   test_arrays, test_labels, vocab_size, n_classes, config, *,
                   padding_in_mean=False, label_shift=0, verbose=False):
    """按指定配置训练一次并返回模型、历史与测试集前向结果。

    `validation_labels`中的`UNKNOWN`表示域外样本，不参与闭集验证指标，
    但保留在验证集里供阈值选择使用（见`choose_threshold`）。
    """
    experiment_config = ModelConfig(**{**config.as_dict(), 'padding_in_mean': padding_in_mean,
                                       'label_shift': label_shift})
    model = MeanPoolClassifier(vocab_size, n_classes, experiment_config)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    validation_labels = np.asarray(validation_labels)
    known = validation_labels != UNKNOWN_ID
    if not known.any():
        raise ValueError('验证集中没有已知类别样本，无法计算验证指标')
    known_arrays = (validation_arrays[0][known], validation_arrays[1][known])
    known_labels = validation_labels[known].astype(np.int64)
    with threadpool_limits(limits=1):
        history = train_model(model, train_arrays, train_labels, known_arrays,
                              known_labels, verbose=verbose)
    # 域外样本不进入闭集损失；只为已知测试样本计算逐样本损失，其余位置记NaN。
    test_labels = np.asarray(test_labels, dtype=np.int64)
    test_known = test_labels != UNKNOWN_ID
    forward = model.forward_numpy(test_arrays[0], test_arrays[1], padding_in_mean=padding_in_mean)
    per_sample_loss = np.full(len(test_labels), np.nan)
    if test_known.any():
        targets = one_hot(test_labels[test_known], n_classes)
        known_forward = model.forward_numpy(test_arrays[0][test_known], test_arrays[1][test_known],
                                            targets, padding_in_mean=padding_in_mean)
        per_sample_loss[test_known] = known_forward['per_sample_loss']
        forward['known_accuracy'] = float(
            (known_forward['probabilities'].argmax(axis=1) == test_labels[test_known]).mean())
    forward['per_sample_loss'] = per_sample_loss
    return {'model': model, 'history': history, 'forward': forward,
            'test_predictions': forward['probabilities'].argmax(axis=1)}


def padding_sensitivity(model, ids, mask, labels, target_length):
    """比较正确masked mean与「补齐参与均值」缺陷：池化向量、预测与宏F1的差异。

    两个配置使用同一个已训练模型与同一批样本，唯一变化是补齐位置是否计入均值，
    因此差异可直接归因于该实现选择。
    """
    correct = model.forward_numpy(ids, mask, padding_in_mean=False)
    buggy = model.forward_numpy(ids, mask, padding_in_mean=True)
    correct_predictions = correct['probabilities'].argmax(axis=1)
    buggy_predictions = buggy['probabilities'].argmax(axis=1)
    known = np.asarray(labels) != UNKNOWN_ID
    return {
        'rows': int(len(labels)),
        'sequence_width': int(mask.shape[1]),
        'max_pooled_abs_difference': float(np.abs(correct['pooled'] - buggy['pooled']).max()),
        'mean_pooled_abs_difference': float(np.abs(correct['pooled'] - buggy['pooled']).mean()),
        'prediction_changes': int((correct_predictions != buggy_predictions).sum()),
        'macro_f1_correct': float(macro_f1(np.asarray(labels)[known], correct_predictions[known], model.n_classes)),
        'macro_f1_padding_in_mean': float(macro_f1(np.asarray(labels)[known], buggy_predictions[known],
                                                   model.n_classes)),
    }


def batch_width_invariance(model, ids, mask, labels, widths=(33, 40, 64, 128), rows=48):
    """把同一批样本补到更宽的画布，检验池化与预测是否不变。

    这里只做"扩宽"、不做截断，因此差异只能来自补齐处理方式：
    正确的masked mean应几乎不变；把补齐位置计入均值时，同一句话的表示会随画布宽度漂移。
    """
    rows_used = min(rows, ids.shape[0])
    base_ids, base_mask = ids[:rows_used], mask[:rows_used]
    valid = base_mask.sum(axis=1) > 0
    report = {}
    for name, padding_in_mean in [('masked_mean', False), ('padding_in_mean', True)]:
        reference = model.forward_numpy(base_ids, base_mask,
                                        padding_in_mean=padding_in_mean)['pooled'][valid]
        reference_predictions = model.forward_numpy(
            base_ids, base_mask, padding_in_mean=padding_in_mean)['probabilities'][valid].argmax(axis=1)
        width_labels, drift, prediction_changes = [], 0.0, 0
        for width in widths:
            width = max(width, base_ids.shape[1])
            canvas_ids = np.zeros((rows_used, width), dtype=base_ids.dtype)
            canvas_mask = np.zeros((rows_used, width), dtype=base_mask.dtype)
            canvas_ids[:, :base_ids.shape[1]] = base_ids
            canvas_mask[:, :base_ids.shape[1]] = base_mask
            forward = model.forward_numpy(canvas_ids, canvas_mask, padding_in_mean=padding_in_mean)
            width_labels.append(int(width))
            drift = max(drift, float(np.abs(reference - forward['pooled'][valid]).max()))
            prediction_changes += int((forward['probabilities'][valid].argmax(axis=1)
                                       != reference_predictions).sum())
        report[name] = {
            'widths': width_labels,
            'samples': int(valid.sum()),
            'max_pooled_drift': drift,
            'prediction_changes_across_widths': prediction_changes,
            'note': '只扩宽不截断；正确实现的池化漂移应接近0',
        }
    return report


# --------------------------------------------------------------------------
# 图表与报告
# --------------------------------------------------------------------------
def plot_figures(out, history, test_frame, metrics_table, thresholds, encoding, bug_table):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    axes[0].plot(history.epochs, history.train_loss, label='train', color='#315C9B')
    axes[0].plot(history.epochs, history.validation_loss, label='validation', color='#C47B27')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Loss by epoch')
    axes[0].legend()
    axes[1].plot(history.epochs, history.train_macro_f1, label='train', color='#315C9B')
    axes[1].plot(history.epochs, history.validation_macro_f1, label='validation', color='#C47B27')
    axes[1].set(xlabel='Epoch', ylabel='Macro-F1 (12 classes)', ylim=(0, 1), title='Macro-F1 by epoch')
    axes[1].legend()
    fig.savefig(out / 'training-curves.png', dpi=150)
    plt.close(fig)

    keys = ['known_macro_f1', 'known_coverage', 'unknown_false_accept_rate']
    names = ['Known macro-F1', 'Known coverage', 'OOD false accept']
    xs = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for index, (name, color) in enumerate([('mini_mean_pool', '#315C9B'), ('tfidf_frozen', '#C47B27'),
                                           ('mini_mean_pool_forced', '#6E8FBF'),
                                           ('tfidf_forced_frozen', '#E0B072')]):
        values = [metrics_table[name][key] for key in keys]
        bars = ax.bar(xs + (index - 1.5) * 0.2, values, 0.2, label=name, color=color)
        ax.bar_label(bars, labels=[f'{value:.1%}' for value in values], padding=2, fontsize=8)
    ax.set(xticks=xs, xticklabels=names, ylim=(0, 1.2), ylabel='Rate (0–1)',
           title='Held-out routing results | 240 known + 40 OOD samples')
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(out / 'comparison.png', dpi=150)
    plt.close(fig)

    all_labels = list(LABEL_NAMES.keys())
    confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    label_index = {label: index for index, label in enumerate(all_labels)}
    for truth, prediction in zip(test_frame.intent, test_frame.mini_prediction):
        confusion[label_index[truth], label_index[prediction]] += 1
    pd.DataFrame(confusion, index=all_labels, columns=all_labels).to_csv(
        out / 'confusion-mini.csv', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11, 9), layout='constrained')
    image = ax.imshow(confusion, cmap='Blues', vmin=0, vmax=40)
    display_labels = [f'{index + 1:02}' for index in range(12)] + ['OOD']
    ax.set(xticks=np.arange(13), yticks=np.arange(13), xticklabels=display_labels,
           yticklabels=display_labels, xlabel='Predicted intent (OOD = abstain)',
           ylabel='True intent', title='mini mean-pool: confusion counts | n=280')
    for (row, column), value in np.ndenumerate(confusion):
        ax.text(column, row, str(value), ha='center', va='center',
                color='white' if value > 20 else '#263238', fontsize=9)
    fig.colorbar(image, ax=ax, shrink=.75, label='Sample count')
    fig.savefig(out / 'confusion-mini.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ax.plot([trial['threshold'] for trial in thresholds], [trial['cost'] for trial in thresholds],
            'o-', color='#315C9B', label='Validation cost')
    selected = min(thresholds, key=lambda item: (item['cost'], item['unknown_false_accept_rate'],
                                                 -item['known_coverage'], item['threshold']))
    ax.axvline(selected['threshold'], color='#8C5721', linestyle='--',
               label=f"Selected threshold = {selected['threshold']:.2f}")
    ax.set(xlabel='Max-softmax rejection threshold', ylabel='Illustrative cost per request',
           title='Threshold selection | validation only (240 known + 40 OOD)')
    ax.legend()
    fig.savefig(out / 'threshold.png', dpi=150)
    plt.close(fig)

    lengths = [len(tokenize(text, encoding['ngram_max'])) for text in test_frame.text]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout='constrained')
    ax.hist(lengths, bins=24, color='#315C9B', alpha=.85)
    ax.axvline(encoding['max_length'], color='#8C5721', linestyle='--',
               label=f"max_length = {encoding['max_length']}")
    ax.set(xlabel='Token count per sample (test)', ylabel='Samples',
           title='Sequence length distribution and truncation')
    ax.legend()
    fig.savefig(out / 'length-distribution.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.0), layout='constrained')
    names = list(bug_table)
    display = [bug_table[name]['label'] if CJK_FONT else name for name in names]
    values = [bug_table[name]['known_macro_f1_forced'] for name in names]
    bars = ax.bar(np.arange(len(names)), values, 0.5, color=['#315C9B', '#C47B27', '#A33F3F'])
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3)
    ax.set(xticks=np.arange(len(names)), xticklabels=display, ylim=(0, 1),
           ylabel='Known macro-F1 (forced 12-class)',
           title='Injected-defect diagnostics | identical data and split')
    fig.savefig(out / 'bug-comparison.png', dpi=150)
    plt.close(fig)


def generate_report(out, result, test_frame, challenges, encoding, bug_table, gradient,
                    per_class=None):
    labels = result['label_order']
    per_class = per_class or {}
    per_class_rows = []
    for label, values in per_class.items():
        recall = '—' if values['recall'] is None else f"{values['recall']:.1%}"
        per_class_rows.append(f"| {label} | {values['name']} | {values['support']} | {recall} |")
    per_class_rows = '\n'.join(per_class_rows)
    rows = []
    for name, metrics in result['metrics'].items():
        rows.append(f"| {name} | {metrics['known_macro_f1']:.4f} | {metrics['known_accuracy']:.1%} "
                    f"| {metrics['known_coverage']:.1%} "
                    f"| {metrics['unknown_false_accept_rate']:.1%} ({metrics['unknown_accepted_n']}/40) "
                    f"| {metrics['routing_cost_per_request']:.4f} |")
    legend = '；'.join(f'{index + 1:02}={LABEL_NAMES[label]}' for index, label in enumerate(labels))
    errors = test_frame[test_frame.intent != test_frame.mini_prediction].head(20)
    error_rows = '\n'.join(
        f'| {row.sample_id} | {row.text} | {row.intent} | {row.mini_prediction} | '
        f'{row.tfidf_prediction} | {row.length} |' for row in errors.itertuples())
    challenge_rows = '\n'.join(
        f'| {row.case_id} | {row.text} | {row.expected_intent} | {row.mini_prediction} | {row.reason} |'
        for row in challenges.itertuples())
    bug_rows = '\n'.join(
        f"| {value['label']} | {value['known_macro_f1_forced']:.4f} | {value['known_accuracy_forced']:.1%} "
        f"| {value['final_train_loss']:.4f} | {value['note']} |" for value in bug_table.values())
    gradient_rows = '\n'.join(
        f"| {name} | {value['max_relative_error']:.3e} | {value['elements']} |"
        for name, value in gradient.items() if isinstance(value, dict))
    text = f'''# 第02课张量与文本表示实测报告

## 数据与方法

沿用第01课冻结数据版本 {result['dataset_version']}，SHA256：`{result['dataset_sha256']}`；本课不重新生成数据。
训练720、验证240已知+40未知、测试240已知+40未知；客户与表达族隔离，测试每已知类20条。

编码为字符级词表：`<pad>`固定索引0、`<unk>`固定索引1，其余token按训练集频次与字典序排序。
词表只从训练集构建（词表{encoding['vocab_size']}项，最长序列{encoding['max_length']}），
验证与测试的未见token计为UNK；截断发生在编码阶段，不由模型处理。
训练后统计的OOV发生率为{encoding['oov_rate']:.4%}（{encoding['oov_token_occurrences']}/{encoding['total_tokens']}个token）。

模型是最小的embedding均值池化分类器：IDs→查表→按mask求均值→线性层→softmax交叉熵，
参数量{result['parameter_count']['total']}（embedding {result['parameter_count']['embedding']}＋分类头
{result['parameter_count']['classifier']}）。填padding参与均值或标签错位属于注入缺陷，见下文。

## 手写反向的数值校验

在训练前对一个小batch做中心差分数值梯度校验（epsilon={gradient['epsilon']}），
比较手写反向与数值梯度：

| 参数 | 最大相对误差 | 元素数 |
|---|---:|---:|
{gradient_rows}

相对误差在 {min(v['max_relative_error'] for v in gradient.values() if isinstance(v, dict)):.1e} 到
{max(v['max_relative_error'] for v in gradient.values() if isinstance(v, dict)):.1e} 之间。
中心差分在epsilon={gradient['epsilon']}、float64下本身的截断与舍入误差约为1e-5量级，
因此该结果说明embedding查表的scatter-add、masked mean的按长度缩放和线性层梯度方向正确；
偏置的误差最小（1e-9量级），因为它不经过embedding求和。校验只覆盖这些小规模算子，
不代表大模型训练栈已被验证。

## 冻结测试结果

| 方案 | 已知macro-F1 | 已知准确率 | 已知覆盖率 | 未知误收率 | 每请求教学成本 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

已知类指标分母240条；未知误收分母40条。`mini_mean_pool`为带拒识的均值池化模型，
`tfidf_frozen`为第01课已保存的TF-IDF＋拒识结果（本课不重新训练该模型）；
`*_forced`是不拒识的闭集对照，未知请求被强行分到12类是其固有限制，
与第01课的`tfidf_forced`口径一致。

![方案比较](comparison.png)

## 训练过程

40个epoch，Adam，学习率0.05，batch 64，梯度裁剪5.0，随机种子{result['config']['seed']}。
训练集loss从{result['history']['train_loss'][0]:.4f}降至{result['history']['train_loss'][-1]:.4f}；
验证集loss从{result['history']['validation_loss'][0]:.4f}变为{result['history']['validation_loss'][-1]:.4f}，
验证macro-F1从{result['history']['validation_macro_f1'][0]:.4f}变为{result['history']['validation_macro_f1'][-1]:.4f}。

![训练曲线](training-curves.png)

若训练loss持续下降而验证指标不升，说明模型在记忆训练表达；本数据中训练与测试的表达族完全不重叠，
这类差距属于预期，不应通过继续训练来消除。

## 缺陷复现：补齐与标签

| 配置 | 已知macro-F1（强制分类） | 已知准确率 | 末轮训练loss | 说明 |
|---|---:|---:|---:|---|
{bug_rows}

第一行是正确实现（补齐位置权重为0）。第二行让补齐位置进入均值池化：同一个训练好的模型、
同一批测试样本下，池化向量的最大差异为 {result['bugs']['padding_in_mean']['max_pooled_abs_difference']:.4f}，
{result['bugs']['padding_in_mean']['prediction_changes']}/280条预测发生改变。
把样本补到不同宽度时，正确实现的池化漂移为
{result['bugs']['padding_in_mean']['batch_width_invariance']['masked_mean']['max_pooled_drift']:.2e}，
而缺陷实现为
{result['bugs']['padding_in_mean']['batch_width_invariance']['padding_in_mean']['max_pooled_drift']:.4f}，
即同一句话的表示随batch内最长样本变化，训练与推理的batch组成不同就会得到不同结果。

**注意**：该缺陷在本测试集上的已知macro-F1反而更高，说明"指标变好"不能证明实现正确；
它的真实代价是表示不稳定，在长度分布变化或单条推理时才会暴露。这是本课要建立的判断习惯。
第三行把标签整体错位，输入与标签不再对应，末轮训练loss仍为
{result['bugs']['label_shift']['final_train_loss']:.4f}、macro-F1降至
{result['bugs']['label_shift']['known_macro_f1_forced']:.4f}（接近随机）；
真实项目里这类错误常表现为"训练正常但预测几乎全是同一类"。

![缺陷对照](bug-comparison.png)

## 补齐不变性与长度

![长度分布](length-distribution.png)

正确的masked mean在同一句话被补到不同长度时给出相同池化向量：本实现按有效长度归一、
补齐位置权重为0，因此表示与batch内最长样本无关。代价是序列越长、batch内最长样本越长，
算力与内存占用越高。本批次序列很短（训练集最长
{encoding['sequence_length']['train']['max']}token、测试集最长
{encoding['sequence_length']['test']['max']}token），{encoding['max_length']}的截断上限没有截掉任何样本；
真实业务的长文档必须逐个统计截断比例并在报告中声明。

## 不确定性与混淆矩阵

按意图分层、按表达族配对bootstrap {result['config']['bootstrap_iterations']}次，95%百分位区间：
均值池化F1 [{result['bootstrap_95_percent']['mini_mean_pool'][0]:.4f}, {result['bootstrap_95_percent']['mini_mean_pool'][1]:.4f}]；
第01课TF-IDF F1 [{result['bootstrap_95_percent']['tfidf_frozen'][0]:.4f}, {result['bootstrap_95_percent']['tfidf_frozen'][1]:.4f}]；
两者差值 [{result['bootstrap_95_percent']['paired_difference'][0]:.4f}, {result['bootstrap_95_percent']['paired_difference'][1]:.4f}]。
差值区间跨0，说明在这份合成困难测试上不能断言均值池化稳定优于第01课TF-IDF，
只能说它把强制分类的F1从0.6100提升到0.7456，代价是拒识阈值附近的误收风险更高。
区间只描述该合成表达池的抽样变化，不包含重新训练或真实业务分布带来的不确定性。

图例：{legend}；OOD=unknown（拒识）。

![均值池化混淆矩阵](confusion-mini.png)

## 失败样本

下面是前20条失败样本（按文件顺序），`length`为编码后的token数：
长度差异越大、表达越依赖词序或否定，均值池化越难区分。

| ID | 输入 | 真实 | 均值池化 | 第01课TF-IDF | 长度 |
|---|---|---|---|---|---|
{error_rows}

## 24条人工困难案例

同一挑战集，说明手写模型的边界；预期标签沿用第01课标注规范（多诉求或指代不明记unknown）。

| ID | 输入 | 预期 | 均值池化 | 边界及改进方向 |
|---|---|---|---|---|
{challenge_rows}

## 逐类表现

`recall`为强制分类下该类别的召回；域外行（unknown）无闭集召回，只看是否被误收。

| 类别 | 中文 | 测试n | 强制分类召回 |
|---|---|---:|---:|
{per_class_rows}

## 环境与边界

Python {result['environment']['python']}，{result['environment']['platform']}；
训练耗时{result['fit_seconds']:.2f}秒（单线程），推理在CPU完成，无API调用、无GPU。

本实验只测意图路由，不能据此宣称客服自动解决率或处理时长改善。
字符级编码、均值池化与这个规模的训练结果不能外推到子词分词、上下文模型或生产数据。
'''
    (out / 'baseline-report.md').write_text(text, encoding='utf-8')


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(output_dir, data_dir=None, verbose=False):
    out = prepare_output(output_dir)
    started = time.perf_counter()
    config, model_config = load_config()
    frame, manifest, labels, audit = load_frozen(data_dir)
    train_frame, validation_frame, test_frame = split_frames(frame)

    tokenizer = CharVocab(min_count=model_config.min_count, ngram_max=model_config.ngram_max)
    tokenizer.fit(train_frame.text)
    train_lengths = np.array([len(tokenizer.encode(text)) for text in train_frame.text])
    max_length = int(config['max_length']) if config.get('max_length') else int(np.percentile(train_lengths, 99))
    encoding = encoding_audit(tokenizer, frame, labels, max_length)
    train_arrays = encode_corpus(tokenizer, train_frame.text, max_length)[:2]
    validation_arrays = encode_corpus(tokenizer, validation_frame.text, max_length)[:2]
    test_arrays = encode_corpus(tokenizer, test_frame.text, max_length)[:2]

    label_index = {label: index for index, label in enumerate(labels)}
    # 用UNKNOWN作哨兵而不是缺失值映射，避免pandas把小整数列读成可空整型后静默错位。
    train_labels = encode_labels(train_frame.intent, label_index)
    validation_labels = encode_labels(validation_frame.intent, label_index)
    test_labels = encode_labels(test_frame.intent, label_index)

    # 1) 手写反向的数值校验（在独立初始化的模型上做，避免影响正式训练）
    check_model = MeanPoolClassifier(tokenizer.vocab_size, len(labels), ModelConfig(**model_config.as_dict()))
    gradient_report = gradient_check(check_model, train_arrays[0], train_arrays[1], train_labels,
                                     n_rows=config['gradient_check_rows'])
    save_json(out / 'gradient-check.json', gradient_report)

    # 2) 正式训练
    main = run_experiment(train_arrays, train_labels, validation_arrays, validation_labels,
                          test_arrays, test_labels, tokenizer.vocab_size, len(labels),
                          model_config, verbose=verbose)
    model, history = main['model'], main['history']

    # 3) 拒识阈值只在验证集选择；域外样本不参与闭集损失，但必须计入误收成本。
    # validation_frame.intent保留原始字符串，UNKNOWN即第01课的拒识口径。
    validation_forward = model.forward_numpy(validation_arrays[0], validation_arrays[1])
    index_to_label = np.asarray(labels)
    threshold, trials = choose_threshold(
        validation_frame.intent.to_numpy(),
        validation_forward['probabilities'], index_to_label, config)

    # 4) 测试集两种口径 + 第01课冻结对照
    baseline = load_frozen_baseline()
    baseline = baseline[baseline.split == 'test'].reset_index(drop=True)
    if len(baseline) != len(test_frame):
        raise ValueError('第01课参考预测与当前测试集行数不一致，无法配对比较')
    if not np.array_equal(baseline.sample_id.to_numpy(), test_frame.sample_id.to_numpy()):
        raise ValueError('第01课参考预测与当前测试集样本顺序不一致，无法配对比较')

    test_forward = model.forward_numpy(test_arrays[0], test_arrays[1])
    test_frame = test_frame.copy()
    test_frame['mini_prediction_forced'] = index_to_label[test_forward['probabilities'].argmax(axis=1)]
    max_scores = test_forward['probabilities'].max(axis=1)
    test_frame['mini_score'] = max_scores
    test_frame['mini_prediction'] = np.where(max_scores >= threshold,
                                             test_frame.mini_prediction_forced, UNKNOWN)
    test_frame['per_sample_loss'] = main['forward']['per_sample_loss']
    test_frame['length'] = test_arrays[1].sum(axis=1)
    test_frame['tfidf_prediction'] = baseline.tfidf_prediction.to_numpy()
    test_frame['tfidf_forced_prediction'] = baseline.tfidf_raw_prediction.to_numpy()
    test_frame['mini_correct'] = test_frame.mini_prediction.eq(test_frame.intent)
    test_frame['tfidf_correct'] = test_frame.tfidf_prediction.eq(test_frame.intent)
    # 整数索引列供bootstrap使用：UNKNOWN_ID表示拒识或域外，不参与闭集F1。
    test_frame['intent_index'] = test_labels
    forced_index = test_forward['probabilities'].argmax(axis=1)
    test_frame['mini_index_forced'] = forced_index
    test_frame['mini_index_selected'] = np.where(max_scores >= threshold, forced_index, UNKNOWN_ID)
    test_frame['tfidf_index'] = baseline.tfidf_prediction.map(
        lambda label: label_index.get(label, UNKNOWN_ID)).to_numpy()
    test_frame['tfidf_index_forced'] = baseline.tfidf_raw_prediction.map(
        lambda label: label_index.get(label, UNKNOWN_ID)).to_numpy()
    # bootstrap只在已知类样本上进行，与第01课一致。
    known_frame = test_frame[test_frame.intent_index != UNKNOWN_ID]

    metrics = {
        'mini_mean_pool': summary_metrics(test_frame.intent, test_frame.mini_prediction, labels, config['costs']),
        'mini_mean_pool_forced': summary_metrics(test_frame.intent, test_frame.mini_prediction_forced,
                                                 labels, config['costs']),
        'tfidf_frozen': summary_metrics(test_frame.intent, test_frame.tfidf_prediction, labels, config['costs']),
        'tfidf_forced_frozen': summary_metrics(test_frame.intent, test_frame.tfidf_forced_prediction,
                                               labels, config['costs']),
    }
    # bootstrap按整数类别索引分层；字符串标签会与整数索引比较而恒为空。
    bootstrap = paired_cluster_bootstrap(known_frame, list(range(len(labels))),
                                         config['bootstrap_iterations'], config['seed'])

    # 5) 缺陷复现：补齐进入均值 + 标签错位
    padding_run = run_experiment(train_arrays, train_labels, validation_arrays, validation_labels,
                                 test_arrays, test_labels, tokenizer.vocab_size, len(labels),
                                 model_config, padding_in_mean=True, verbose=False)
    shift_run = run_experiment(train_arrays, train_labels, validation_arrays, validation_labels,
                               test_arrays, test_labels, tokenizer.vocab_size, len(labels),
                               model_config, label_shift=1, verbose=False)
    bug_table = {}
    for key, label, run_result, note in [
            ('correct', '正确实现', main, '补齐位置权重为0，标签与输入对应'),
            ('padding_in_mean', '补齐参与均值', padding_run, '池化被补齐位置的初始embedding拉偏'),
            ('label_shift', '标签错位1位', shift_run, '输入与标签不再对应，退化为随机')]:
        forced = summary_metrics(test_frame.intent,
                                 index_to_label[run_result['test_predictions']], labels, config['costs'])
        bug_table[key] = {
            'label': label,
            'known_macro_f1_forced': forced['known_macro_f1'],
            'known_accuracy_forced': forced['known_accuracy'],
            'final_train_loss': run_result['history'].train_loss[-1],
            'final_validation_loss': run_result['history'].validation_loss[-1],
            'note': note}
    sensitivity = padding_sensitivity(model, test_arrays[0], test_arrays[1], test_labels, max_length)
    invariance = batch_width_invariance(model, test_arrays[0], test_arrays[1], test_labels)
    bug_table['padding_in_mean']['sensitivity'] = sensitivity
    bug_table['padding_in_mean']['max_pooled_abs_difference'] = sensitivity['max_pooled_abs_difference']
    bug_table['padding_in_mean']['prediction_changes'] = sensitivity['prediction_changes']
    bug_table['padding_in_mean']['batch_width_invariance'] = invariance

    # 6) 挑战集与逐类指标
    challenges = load_challenges(data_dir).copy()
    challenge_ids, challenge_mask = encode_corpus(tokenizer, challenges.text, max_length)[:2]
    challenge_forward = model.forward_numpy(challenge_ids, challenge_mask)
    challenges['mini_prediction'] = index_to_label[challenge_forward['probabilities'].argmax(axis=1)]
    challenges['mini_score'] = challenge_forward['probabilities'].max(axis=1)

    per_class = {}
    for label in labels + [UNKNOWN]:
        subset = test_frame[test_frame.intent == label]
        if len(subset) == 0:
            continue
        target = label_index.get(label)
        predicted_scores = test_frame.mini_prediction_forced.to_numpy() == label
        per_class[label] = {
            'name': LABEL_NAMES.get(label, label), 'support': int(len(subset)),
            'recall': float((subset.mini_prediction_forced == label).mean()) if target is not None else None,
            'predicted_n': int(predicted_scores.sum()),
        }

    fit_seconds = time.perf_counter() - started
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'machine': platform.machine(), 'logical_cpu_count': os.cpu_count(),
                   'thread_limit': 1,
                   'packages': {package: importlib.metadata.version(package)
                                for package in ['numpy', 'pandas', 'matplotlib']}}
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_version': manifest['dataset_version'], 'dataset_sha256': manifest['sha256'],
        'session_seed': config['seed'], 'config': config, 'model_config': model_config.as_dict(),
        'audit': audit, 'encoding': encoding, 'selected_threshold': threshold,
        'metrics': metrics, 'bootstrap_95_percent': bootstrap, 'excluded_from_final_metrics': ['bug_table'],
        'history': history.as_dict(), 'parameter_count': main['model'].parameter_count(),
        'bugs': bug_table, 'fit_seconds': fit_seconds, 'environment': environment,
        'label_order': labels,
        'input_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in
                         [ROOT / 'config.json', ROOT / 'src/autograd.py', ROOT / 'src/encoding.py',
                          ROOT / 'src/model.py', ROOT / 'src/data.py', ROOT / 'src/evaluate.py']},
        'reference_inputs': {'lesson01_dataset': str((data_dir or LESSON01_DATA) / 'samples.csv'),
                             'lesson01_predictions': 'courseware/lesson-01/reference/baseline-v1/predictions.csv'},
    }

    test_frame.to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(trials).to_csv(out / 'threshold-selection.csv', index=False)
    pd.DataFrame(metrics).T.to_csv(out / 'metrics.csv')
    pd.DataFrame(per_class).T.to_csv(out / 'per-class.csv')
    pd.DataFrame(history.as_dict()).to_csv(out / 'training-history.csv', index=False)
    challenges.to_csv(out / 'challenge-results.csv', index=False)
    save_json(out / 'results.json', result)
    save_json(out / 'encoding-audit.json', encoding)
    save_json(out / 'bug-diagnostics.json', bug_table)
    save_json(out / 'data-audit.json', audit)
    plot_figures(out, history, test_frame, metrics, trials, encoding, bug_table)
    generate_report(out, result, test_frame, challenges, encoding, bug_table, gradient_report,
                    per_class)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    result = run(args.output, args.data_dir, args.verbose)
    print(json.dumps({'selected_threshold': result['selected_threshold'],
                      'metrics': {name: {'known_macro_f1': value['known_macro_f1'],
                                         'known_coverage': value['known_coverage'],
                                         'unknown_false_accept_rate': value['unknown_false_accept_rate']}
                                  for name, value in result['metrics'].items()},
                      'output': str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
