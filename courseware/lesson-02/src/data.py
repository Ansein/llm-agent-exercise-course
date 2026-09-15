"""读取第01课冻结的客服意图数据，只做校验与切分，不重新生成、不覆盖。

课程约定：第02课沿用同一份冻结数据与同一测试集，才能与第01课基线公平对照。
数据文件属于第01课教学包，本课只读引用；若需修改数据，应新建数据集版本。
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LESSON01_DATA = ROOT.parent / 'lesson-01' / 'data'
LESSON01_REFERENCE = ROOT.parent / 'lesson-01' / 'reference' / 'baseline-v1'
UNKNOWN = 'unknown'
LABEL_NAMES = {
    'shipping_status': '物流查询', 'cancel_order': '取消订单', 'refund_status': '退款进度',
    'return_request': '退货申请', 'exchange_request': '换货申请', 'invoice': '发票问题',
    'payment': '支付问题', 'change_address': '修改收件信息', 'warranty': '保修维修',
    'technical_help': '安装使用', 'complaint': '服务投诉', 'product_info': '售前咨询',
    UNKNOWN: '拒识/未知'}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(data_dir=None):
    """校验冻结hash后读取数据，并核对数量、分组隔离与标签集合。"""
    data_dir = Path(data_dir) if data_dir is not None else LESSON01_DATA
    if not (data_dir / 'samples.csv').is_file():
        raise FileNotFoundError(
            f'未找到第01课冻结数据 {data_dir}/samples.csv；本课不重新生成数据，'
            '请先确认第01课教学包完整')
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    observed = sha256(data_dir / 'samples.csv')
    if observed != manifest['sha256']:
        raise ValueError(f'冻结数据hash不匹配：期望{manifest["sha256"]}，实际{observed}；'
                         '不要覆盖冻结数据，应新建数据集版本')
    for name, digest in manifest['source_hashes'].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f'源文件已被修改但未更新数据集版本：{name}')
    frame = pd.read_csv(data_dir / 'samples.csv', keep_default_na=False)
    labels = [label for label in json.loads((data_dir / 'templates.json').read_text(encoding='utf-8'))]
    audit = audit_frame(frame, labels)
    return frame, manifest, labels, audit


def audit_frame(frame, labels) -> dict:
    """与第01课一致的隔离检查：客户与表达族不得跨集合，且测试类别均衡。"""
    required = {'sample_id', 'customer_id', 'template_id', 'text', 'intent', 'split',
                'source', 'dataset_version'}
    if not required <= set(frame.columns):
        raise ValueError(f'缺少字段: {required - set(frame.columns)}')
    if frame[list(required)].isna().any().any() or frame.text.str.strip().eq('').any():
        raise ValueError('存在空值或空文本')
    if not frame.sample_id.is_unique:
        raise ValueError('sample_id重复')
    if not set(frame.intent) <= set(labels + [UNKNOWN]):
        raise ValueError('出现词表外的意图标签')
    if not set(frame.split) <= {'train', 'validation', 'test'}:
        raise ValueError('非法split取值')
    for key in ['customer_id', 'template_id']:
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f'{key}跨集合泄漏')
    known = frame[frame.intent != UNKNOWN]
    test_by_label = known[known.split == 'test'].groupby('intent').size()
    if not (test_by_label == 20).all():
        raise ValueError(f'已知测试类不均衡: {test_by_label.to_dict()}')
    train = frame[frame.split == 'train']
    if (train.intent == UNKNOWN).any():
        raise ValueError('未知意图不得进入闭集训练')
    return {
        'rows': int(len(frame)),
        'known_rows': int(len(known)),
        'unknown_rows': int((frame.intent == UNKNOWN).sum()),
        'split_counts': {key: int(value) for key, value in frame.split.value_counts().items()},
        'known_test_rows': int(len(known[known.split == 'test'])),
        'unknown_test_rows': int(((frame.intent == UNKNOWN) & (frame.split == 'test')).sum()),
        'customer_groups': int(frame.customer_id.nunique()),
        'template_groups': int(frame.template_id.nunique()),
        'test_label_counts': {key: int(value) for key, value in sorted(test_by_label.items())},
        'label_order': list(labels),
    }


def split_frames(frame):
    """返回训练、验证、测试三个子集，保持原始行序。"""
    return (frame[frame.split == 'train'].reset_index(drop=True),
            frame[frame.split == 'validation'].reset_index(drop=True),
            frame[frame.split == 'test'].reset_index(drop=True))


def load_challenges(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else LESSON01_DATA
    return pd.read_csv(data_dir / 'challenges.csv', keep_default_na=False)


def load_frozen_baseline(reference_dir=None, name='predictions.csv'):
    """读取第01课已保存的逐样本预测，用于配对比较，不重新训练其模型。"""
    reference_dir = Path(reference_dir) if reference_dir is not None else LESSON01_REFERENCE
    path = reference_dir / name
    if not path.is_file():
        raise FileNotFoundError(f'未找到第01课参考输出 {path}；无法进行配对对照')
    return pd.read_csv(path, keep_default_na=False)
