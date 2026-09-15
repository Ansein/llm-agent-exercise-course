"""字符级文本编码：词表、特殊token、补齐与attention mask。

教学要点：
- 词表只能从训练集构建，验证与测试只允许查表，未见字符映射为UNK。
- 补齐（padding）产生无效位置，必须用mask显式排除，不能依赖模型自行忽略。
- 序列长度是资源：批量越大、序列越长，显存与算力占用越高。
"""
import unicodedata
from collections import Counter

import numpy as np

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN)
PAD_ID = 0
UNK_ID = 1


def normalize(text: str) -> str:
    """NFKC归一化并转小写，保留空白以维持字符序列可读；连续空白压成一个空格。"""
    if not isinstance(text, str):
        raise TypeError('text必须是字符串')
    folded = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(folded.split())


def tokenize(text: str, ngram_max: int = 1) -> list[str]:
    """字符级分词；ngram_max>1时附加2..ngram_max元字符片段。

    返回顺序为：所有单字，随后按阶数递增的字符片段。词表因此同时含字与片段。
    """
    if ngram_max < 1:
        raise ValueError('ngram_max至少为1')
    chars = list(normalize(text))
    tokens = list(chars)
    for n in range(2, ngram_max + 1):
        tokens.extend(''.join(chars[i:i + n]) for i in range(len(chars) - n + 1))
    return tokens


class CharVocab:
    """字符级词表。索引0固定为PAD，索引1固定为UNK，其余按频次与字典序稳定排序。"""

    def __init__(self, min_count: int = 1, ngram_max: int = 1):
        if min_count < 1:
            raise ValueError('min_count至少为1')
        self.min_count = min_count
        self.ngram_max = ngram_max
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.oov_tokens: Counter = Counter()
        self.fitted = False

    def fit(self, texts) -> 'CharVocab':
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text, self.ngram_max))
        kept = [(token, count) for token, count in counts.items() if count >= self.min_count]
        kept.sort(key=lambda pair: (-pair[1], pair[0]))
        self.id_to_token = list(SPECIAL_TOKENS) + [token for token, _ in kept]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.fitted = True
        return self

    def __len__(self):
        return len(self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """编码为ID序列；超长截断，未见token累计到oov统计。"""
        if not self.fitted:
            raise RuntimeError('必须先在训练文本上调用fit')
        ids = []
        for token in tokenize(text, self.ngram_max):
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(UNK_ID)
                self.oov_tokens[token] += 1
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids) -> str:
        return ''.join(self.id_to_token[int(i)] for i in ids if int(i) not in (PAD_ID, UNK_ID))


def pad_batch(sequences, max_length: int | None = None):
    """把一个batch的ID序列补齐为 (batch, seq)，同时返回 (batch, seq) 的attention mask。

    mask=1表示真实token，mask=0表示补齐位置。长度为0的序列补成1个PAD，mask全0。
    """
    sequences = [list(seq) for seq in sequences]
    if not sequences:
        raise ValueError('空batch无法补齐')
    longest = max((len(seq) for seq in sequences), default=0)
    width = longest if max_length is None else min(max_length, max(longest, 1))
    width = max(width, 1)
    ids = np.zeros((len(sequences), width), dtype=np.int64)
    mask = np.zeros((len(sequences), width), dtype=np.int64)
    for row, seq in enumerate(sequences):
        for column, token_id in enumerate(seq[:width]):
            ids[row, column] = token_id
            mask[row, column] = 1
    return ids, mask


def sequence_lengths(mask) -> np.ndarray:
    """mask行和即有效长度。"""
    return np.asarray(mask, dtype=np.int64).sum(axis=1)


def encode_corpus(tokenizer: CharVocab, texts, max_length: int | None = None):
    """编码整个语料：返回ids、mask与有效长度，供后续分批或整体训练使用。"""
    sequences = [tokenizer.encode(text, max_length) for text in texts]
    ids, mask = pad_batch(sequences, max_length=max_length)
    return ids, mask, sequence_lengths(mask)
