"""最小自动微分引擎：只实现本课需要的运算，用于讲清前向与反向的形状和梯度。

设计边界：
- 仅支持浮点ndarray、标量广播、embedding查表、masked mean、线性层、softmax交叉熵。
- 反向传播为按拓扑序的手写反向累积；没有通用算子覆盖，也没有GPU或图优化。
- 教学用途，不追求与PyTorch等框架的性能或算子兼容性。
"""
import numpy as np

Array = np.ndarray


class Tensor:
    """带梯度的数组。`requires_grad`为True时才累积`grad`。"""

    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, name=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.name = name
        self.grad: Array | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward = lambda: None

    # ---- 构造辅助 ----
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f'Tensor(shape={self.shape}, requires_grad={self.requires_grad}, name={self.name!r})'

    def _accumulate(self, value: Array):
        if not self.requires_grad:
            return
        value = np.asarray(value, dtype=np.float64)
        self.grad = value if self.grad is None else self.grad + value

    def _record(self, parents, backward):
        """登记反向函数；一旦进入计算图，梯度位置先置零，便于安全累加。"""
        if any(parent.requires_grad for parent in parents):
            self.requires_grad = True
            self._parents = tuple(parents)
            self._backward = backward
            if self.grad is None:
                self.grad = np.zeros_like(self.data)


def _ensure_tensor(value) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


# --------------------------------------------------------------------------
# 前向算子
# --------------------------------------------------------------------------
def embedding(table: Tensor, indices: Array, name='embedding') -> Tensor:
    """查表：indices形状为 (batch, seq)，输出形状为 (batch, seq, hidden)。"""
    indices = np.asarray(indices)
    if indices.ndim != 2:
        raise ValueError(f'indices必须是 (batch, seq) 二维，实际为 {indices.shape}')
    if indices.min() < 0 or indices.max() >= table.shape[0]:
        raise ValueError(f'indices越界: [{indices.min()}, {indices.max()}] 不在 [0, {table.shape[0]})')
    if table.data.ndim != 2:
        raise ValueError(f'embedding表必须是 (vocab, hidden) 二维，实际为 {table.shape}')
    rows = table.data[indices]
    out = Tensor(rows, requires_grad=table.requires_grad, name=name)

    def backward():
        grad = np.zeros_like(table.data)
        np.add.at(grad, indices.reshape(-1), out.grad.reshape(-1, table.shape[1]))
        table._accumulate(grad)

    out._record((table,), backward)
    return out


def masked_mean(embedded: Tensor, mask: Array, name='pooled') -> Tensor:
    """按mask对sequence维求均值，形状 (batch, seq, hidden) -> (batch, hidden)。

    mask为1的位置参与平均；全0行返回0向量，避免除零。
    """
    embedded_arr, mask_arr = embedded.data, np.asarray(mask, dtype=np.float64)
    if embedded_arr.ndim != 3:
        raise ValueError(f'embedded必须是 (batch, seq, hidden) 三维，实际为 {embedded.shape}')
    if mask_arr.shape != embedded_arr.shape[:2]:
        raise ValueError(f'mask形状{mask_arr.shape}与(batch, seq) {embedded_arr.shape[:2]}不一致')
    counts = mask_arr.sum(axis=1, keepdims=True)
    safe = np.where(counts == 0, 1.0, counts)
    pooled = (embedded_arr * mask_arr[:, :, None]).sum(axis=1) / safe
    out = Tensor(pooled, requires_grad=embedded.requires_grad, name=name)

    def backward():
        scaled = out.grad[:, None, :] * mask_arr[:, :, None] / safe[:, :, None]
        embedded._accumulate(scaled)

    out._record((embedded,), backward)
    return out


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None, name='linear') -> Tensor:
    """仿射变换 (batch, in) @ (in, out) + (out,)。"""
    out_data = x.data @ weight.data
    if bias is not None:
        out_data = out_data + bias.data
    parents = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(out_data, requires_grad=any(parent.requires_grad for parent in parents), name=name)

    def backward():
        weight._accumulate(x.data.T @ out.grad)
        x._accumulate(out.grad @ weight.data.T)
        if bias is not None:
            bias._accumulate(out.grad.sum(axis=0))

    out._record(parents, backward)
    return out


def softmax_cross_entropy(logits: Tensor, targets: Array, mask: Array | None = None, name='loss') -> Tensor:
    """softmax交叉熵。`mask`为1的位置计入平均，用于排除补齐位置。

    targets为 (batch, n_classes) one-hot；返回标量平均损失。
    """
    logits_arr = logits.data
    targets_arr = np.asarray(targets, dtype=np.float64)
    if logits_arr.shape != targets_arr.shape:
        raise ValueError(f'logits{logits_arr.shape}与targets{targets_arr.shape}形状不一致')
    if mask is None:
        mask_arr = np.ones(logits_arr.shape[0], dtype=np.float64)
    else:
        mask_arr = np.asarray(mask, dtype=np.float64).reshape(-1)
        if mask_arr.shape[0] != logits_arr.shape[0]:
            raise ValueError(f'mask长度{mask_arr.shape[0]}与batch {logits_arr.shape[0]}不一致')
    denominator = mask_arr.sum()
    if denominator <= 0:
        raise ValueError('mask全为0，损失无定义')

    # 手工log-softmax：减去每行最大值以稳定数值，再减去logsumexp。
    maximum = logits_arr.max(axis=1, keepdims=True)
    shifted = logits_arr - maximum
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    per_sample = -(targets_arr * log_probs).sum(axis=1)
    loss_value = float((per_sample * mask_arr).sum() / denominator)
    out = Tensor([[loss_value]], requires_grad=logits.requires_grad, name=name)

    def backward():
        upstream = float(out.grad.reshape(-1)[0]) / denominator
        probabilities = np.exp(log_probs)
        logits._accumulate(upstream * (probabilities - targets_arr) * mask_arr[:, None])

    out._record((logits,), backward)
    return out


# --------------------------------------------------------------------------
# 反向传播与数值校验
# --------------------------------------------------------------------------
def backward(loss: Tensor) -> None:
    """按拓扑序执行反向传播，把梯度累积到各叶子张量的`grad`。

    这里用入度版拓扑排序（Kahn）：某节点的全部消费者都已入列后才轮到它，
    因此梯度先到输出端、再沿父节点回传。顺序写错时中间梯度会保持0，
    这是自动微分最常见的实现错误之一，本课用数值梯度校验把它暴露出来。
    """
    if loss.data.size != 1:
        raise ValueError('backward只接受标量损失')
    nodes: list[Tensor] = []
    seen: set[int] = set()
    stack = [loss]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        stack.extend(node._parents)
    consumers = {id(node): 0 for node in nodes}
    for node in nodes:
        for parent in node._parents:
            consumers[id(parent)] += 1
    ready = [node for node in nodes if consumers[id(node)] == 0]
    ordered: list[Tensor] = []
    while ready:
        node = ready.pop()
        ordered.append(node)
        for parent in node._parents:
            consumers[id(parent)] -= 1
            if consumers[id(parent)] == 0:
                ready.append(parent)
    if len(ordered) != len(nodes):
        raise RuntimeError('计算图中存在环，无法反向传播')
    # 所有节点梯度清零后从损失出发；损失为标量，其对自身的导数为1。
    for node in ordered:
        node.grad = np.zeros_like(node.data)
    loss.grad = np.ones_like(loss.data)
    for node in ordered:
        node._backward()


def numerical_gradients(loss_fn, tensors, epsilon=1e-6):
    """中心差分数值梯度，用于校验解析反向。tensors为需要求梯度的Tensor列表。"""
    grads = []
    for tensor in tensors:
        grad = np.zeros_like(tensor.data)
        flat = tensor.data.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + epsilon
            plus = loss_fn()
            flat[i] = original - epsilon
            minus = loss_fn()
            flat[i] = original
            grad.reshape(-1)[i] = (plus - minus) / (2 * epsilon)
        grads.append(grad)
    return grads


def relative_error(a: Array, b: Array) -> float:
    """解析梯度与数值梯度的相对误差，分母加极小量避免除零。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))
