"""embedding均值池化分类器：前向、损失、反向与训练循环。

模型刻意做到最小可解释：
    IDs (batch, seq) -> embedding查表 (batch, seq, hidden)
    -> 按mask求均值 (batch, hidden) -> 线性层 (batch, classes) -> softmax交叉熵

参数量 = vocab_size × hidden + hidden × classes + classes，可直接与第01课稀疏词袋对照。
"""
from dataclasses import dataclass, field

import numpy as np

from autograd import Tensor, backward, embedding, linear, masked_mean, softmax_cross_entropy

MIN_COUNT = 1
NGRAM_MAX = 1
HIDDEN_DIM = 64
EMBED_INIT_SCALE = 0.1
EPOCHS = 40
LEARNING_RATE = 0.05
BATCH_SIZE = 64
GRAD_CLIP = 5.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
PADDING_IN_MEAN = False
LABEL_SHIFT = 0


@dataclass
class ModelConfig:
    """一次训练的全部可声明变量；改变其中任一项都应在新runs目录留档。"""
    seed: int = 20260915
    hidden_dim: int = HIDDEN_DIM
    min_count: int = MIN_COUNT
    ngram_max: int = NGRAM_MAX
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    grad_clip: float = GRAD_CLIP
    padding_in_mean: bool = PADDING_IN_MEAN
    label_shift: int = LABEL_SHIFT

    def as_dict(self) -> dict:
        return {'seed': self.seed, 'hidden_dim': self.hidden_dim, 'min_count': self.min_count,
                'ngram_max': self.ngram_max, 'epochs': self.epochs,
                'learning_rate': self.learning_rate, 'batch_size': self.batch_size,
                'grad_clip': self.grad_clip, 'padding_in_mean': self.padding_in_mean,
                'label_shift': self.label_shift}


class MeanPoolClassifier:
    """最小文本分类器：可训练embedding + masked mean + 线性分类头。"""

    def __init__(self, vocab_size: int, n_classes: int, config: ModelConfig):
        self.config = config
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        rng = np.random.default_rng(config.seed)
        scale = EMBED_INIT_SCALE
        self.embedding_table = Tensor(
            rng.normal(0.0, scale, size=(self.vocab_size, config.hidden_dim)), requires_grad=True,
            name='embedding_table')
        self.weight = Tensor(
            rng.normal(0.0, scale, size=(config.hidden_dim, self.n_classes)), requires_grad=True,
            name='classifier_weight')
        self.bias = Tensor(np.zeros(self.n_classes), requires_grad=True, name='classifier_bias')

    @property
    def parameters(self):
        return [self.embedding_table, self.weight, self.bias]

    def parameter_count(self) -> dict:
        return {
            'embedding': int(self.embedding_table.data.size),
            'classifier': int(self.weight.data.size + self.bias.data.size),
            'total': int(sum(parameter.data.size for parameter in self.parameters)),
        }

    def forward(self, ids, mask, *, padding_in_mean: bool | None = None):
        """返回 (logits, pooled, effective_mask)，并构建反向图。

        `padding_in_mean=True`时把补齐位置也计入均值，用于课堂复现缺陷。
        只需要前向数值时用`forward_numpy`，避免无谓地累积梯度。
        """
        use_full = self.config.padding_in_mean if padding_in_mean is None else padding_in_mean
        effective_mask = np.ones_like(mask, dtype=np.int64) if use_full else np.asarray(mask)
        embedded = embedding(self.embedding_table, ids)
        pooled = masked_mean(embedded, effective_mask)
        logits = linear(pooled, self.weight, self.bias)
        return logits, pooled, effective_mask

    def loss(self, ids, mask, targets, *, padding_in_mean: bool | None = None):
        logits, pooled, _ = self.forward(ids, mask, padding_in_mean=padding_in_mean)
        return softmax_cross_entropy(logits, targets), logits, pooled

    def forward_numpy(self, ids, mask, targets=None, *, padding_in_mean: bool | None = None):
        """纯NumPy前向（不建图）：用于历史记录、缺陷复现与逐样本分析。

        返回dict：logits、probabilities、pooled、每样本损失、平均损失与effective_mask。
        """
        use_full = self.config.padding_in_mean if padding_in_mean is None else padding_in_mean
        effective_mask = np.ones_like(mask, dtype=np.int64) if use_full else np.asarray(mask)
        embedded = self.embedding_table.data[ids]
        counts = effective_mask.sum(axis=1, keepdims=True)
        safe = np.where(counts == 0, 1.0, counts)
        pooled = (embedded * effective_mask[:, :, None]).sum(axis=1) / safe
        logits = pooled @ self.weight.data + self.bias.data
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probabilities = np.exp(shifted - log_sum_exp)
        result = {'logits': logits, 'probabilities': probabilities, 'pooled': pooled,
                  'effective_mask': effective_mask}
        if targets is not None:
            targets = np.asarray(targets, dtype=np.float64)
            per_sample = -(targets * (shifted - log_sum_exp)).sum(axis=1)
            result['per_sample_loss'] = per_sample
            result['loss'] = float(per_sample.mean())
        return result

    def predict(self, ids, mask, *, padding_in_mean: bool | None = None):
        """推理：只做前向，返回预测类别索引与softmax概率。"""
        forward = self.forward_numpy(ids, mask, padding_in_mean=padding_in_mean)
        return forward['probabilities'].argmax(axis=1), forward['probabilities']


def _as_label_array(labels, name: str) -> np.ndarray:
    """把标签转成整数数组；遇到缺失值或无法转换的值时明确报错，避免静默转型。"""
    values = np.asarray(labels)
    if values.dtype.kind not in 'iu':
        if values.dtype.kind != 'f' or not np.isfinite(values.astype(np.float64)).all():
            raise ValueError(f'{name}含缺失值或非数值标签；域外样本应先过滤再计算闭集指标')
        values = values.astype(np.int64)
    return values.astype(np.int64, copy=False)


def one_hot(labels, n_classes: int) -> np.ndarray:
    labels = _as_label_array(labels, 'labels')
    if labels.size == 0:
        raise ValueError('标签为空')
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f'标签越界: [{labels.min()}, {labels.max()}] 不在 [0, {n_classes})')
    encoded = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def make_batches(n_rows: int, batch_size: int, rng: np.random.Generator):
    """打乱后按batch_size切分；返回行索引列表。同一seed下顺序完全可复现。"""
    order = rng.permutation(n_rows)
    return [order[start:start + batch_size] for start in range(0, n_rows, batch_size)]


def macro_f1(truth, prediction, n_classes: int) -> float:
    """未加权平均各类F1；无正预测的类记0，与第01课zero_division=0口径一致。"""
    truth = _as_label_array(truth, 'truth')
    prediction = _as_label_array(prediction, 'prediction')
    if truth.shape != prediction.shape:
        raise ValueError(f'truth{truth.shape}与prediction{prediction.shape}长度不一致')
    scores = []
    for label in range(n_classes):
        true_positive = int(((truth == label) & (prediction == label)).sum())
        false_positive = int(((truth != label) & (prediction == label)).sum())
        false_negative = int(((truth == label) & (prediction != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


@dataclass
class TrainingHistory:
    epochs: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    train_macro_f1: list = field(default_factory=list)
    validation_loss: list = field(default_factory=list)
    validation_macro_f1: list = field(default_factory=list)

    def record(self, epoch, train_loss, train_f1, validation_loss, validation_f1):
        self.epochs.append(int(epoch))
        self.train_loss.append(float(train_loss))
        self.train_macro_f1.append(float(train_f1))
        self.validation_loss.append(float(validation_loss))
        self.validation_macro_f1.append(float(validation_f1))

    def as_dict(self) -> dict:
        return {'epochs': self.epochs, 'train_loss': self.train_loss,
                'train_macro_f1': self.train_macro_f1,
                'validation_loss': self.validation_loss,
                'validation_macro_f1': self.validation_macro_f1}


def evaluate_loss_and_f1(model, ids, mask, labels, *, padding_in_mean=None):
    """整批推理计算平均损失与macro-F1，不更新参数、不累积梯度。"""
    targets = one_hot(labels, model.n_classes)
    forward = model.forward_numpy(ids, mask, targets, padding_in_mean=padding_in_mean)
    predictions = forward['probabilities'].argmax(axis=1)
    return float(forward['loss']), macro_f1(labels, predictions, model.n_classes), predictions


def train_model(model, train_arrays, train_labels, validation_arrays, validation_labels,
                *, verbose=False):
    """Adam + 全批/分批梯度下降；返回训练历史。

    标签错位缺陷通过`config.label_shift`实现：把targets整体右移若干位，使输入与标签不再对应。
    """
    config = model.config
    rng = np.random.default_rng(config.seed + 1)
    ids, mask = train_arrays
    shifted = np.roll(train_labels, config.label_shift) if config.label_shift else np.asarray(train_labels)
    history = TrainingHistory()
    moments = [{'m': np.zeros_like(parameter.data), 'v': np.zeros_like(parameter.data)}
               for parameter in model.parameters]
    step = 0
    for epoch in range(1, config.epochs + 1):
        batch_losses = []
        train_predictions = np.zeros(len(train_labels), dtype=np.int64)
        for rows in make_batches(len(train_labels), config.batch_size, rng):
            for parameter in model.parameters:
                parameter.zero_grad()
            targets = one_hot(shifted[rows], model.n_classes)
            loss, _, _ = model.loss(ids[rows], mask[rows], targets)
            backward(loss)
            step += 1
            for parameter, moment in zip(model.parameters, moments):
                gradient = np.clip(parameter.grad, -config.grad_clip, config.grad_clip)
                moment['m'] = ADAM_BETA1 * moment['m'] + (1 - ADAM_BETA1) * gradient
                moment['v'] = ADAM_BETA2 * moment['v'] + (1 - ADAM_BETA2) * gradient ** 2
                corrected_m = moment['m'] / (1 - ADAM_BETA1 ** step)
                corrected_v = moment['v'] / (1 - ADAM_BETA2 ** step)
                parameter.data -= config.learning_rate * corrected_m / (np.sqrt(corrected_v) + ADAM_EPSILON)
            batch_losses.append(float(loss.data.reshape(-1)[0]))
            batch_forward = model.forward_numpy(ids[rows], mask[rows])
            train_predictions[rows] = batch_forward['probabilities'].argmax(axis=1)
        validation_loss, validation_f1, _ = evaluate_loss_and_f1(
            model, validation_arrays[0], validation_arrays[1], validation_labels)
        train_f1 = macro_f1(train_labels, train_predictions, model.n_classes)
        history.record(epoch, float(np.mean(batch_losses)), train_f1, validation_loss, validation_f1)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            print(f'epoch {epoch:02d} | train loss {history.train_loss[-1]:.4f} '
                  f'| train macro-F1 {train_f1:.4f} | validation loss {validation_loss:.4f} '
                  f'| validation macro-F1 {validation_f1:.4f}')
    return history
