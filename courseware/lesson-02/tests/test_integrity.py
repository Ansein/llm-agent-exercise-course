"""第02课回归检查：反向正确性、mask语义、可复现性、数据隔离与参考结果口径。

这些检查针对真实失败模式，而不是覆盖率数字：梯度顺序、补齐参与池化、标签错位、
词表越界、输出覆盖和指标分母错误都会让实验看起来正常但结论不可信。
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from autograd import (Tensor, backward, embedding, linear, masked_mean, numerical_gradients,
                      relative_error, softmax_cross_entropy)
from data import LABEL_NAMES, UNKNOWN, load_challenges, load_frozen, load_frozen_baseline, split_frames
from encoding import PAD_ID, UNK_ID, CharVocab, encode_corpus, normalize, pad_batch, tokenize
from evaluate import (UNKNOWN_ID, batch_width_invariance, encode_labels, gradient_check,
                      padding_sensitivity, paired_cluster_bootstrap, prepare_output, run,
                      summary_metrics)
from model import MeanPoolClassifier, ModelConfig, macro_f1, one_hot, train_model

REFERENCE = ROOT / 'reference' / 'mini-autograd-v1'


class FixtureMixin:
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.manifest, cls.labels, cls.audit = load_frozen()
        cls.train, cls.validation, cls.test = split_frames(cls.frame)
        cls.tokenizer = CharVocab(min_count=1, ngram_max=1).fit(cls.train.text)
        cls.train_arrays = encode_corpus(cls.tokenizer, cls.train.text, 128)[:2]
        cls.validation_arrays = encode_corpus(cls.tokenizer, cls.validation.text, 128)[:2]
        cls.test_arrays = encode_corpus(cls.tokenizer, cls.test.text, 128)[:2]
        cls.label_index = {label: index for index, label in enumerate(cls.labels)}
        cls.train_labels = encode_labels(cls.train.intent, cls.label_index)
        cls.validation_labels = encode_labels(cls.validation.intent, cls.label_index)
        cls.test_labels = encode_labels(cls.test.intent, cls.label_index)


class EncodingTests(FixtureMixin, unittest.TestCase):
    def test_normalize_and_tokenize_are_stable(self):
        self.assertEqual(normalize('  Ａ  B  '), 'a b')
        self.assertEqual(tokenize('退款', 1), ['退', '款'])
        self.assertEqual(tokenize('退款', 2), ['退', '款', '退款'])

    def test_special_token_ids_are_fixed(self):
        self.assertEqual(PAD_ID, 0)
        self.assertEqual(UNK_ID, 1)
        self.assertEqual(self.tokenizer.id_to_token[PAD_ID], '<pad>')
        self.assertEqual(self.tokenizer.id_to_token[UNK_ID], '<unk>')

    def test_vocab_built_from_train_only_and_oov_maps_to_unk(self):
        fresh = CharVocab(min_count=1, ngram_max=1).fit(self.train.text)
        before = fresh.vocab_size
        # 选择训练语料中确定不出现的字符，避免用例依赖具体语料内容。
        unseen = '火星独角兽'
        self.assertTrue(all(character not in fresh.token_to_id for character in unseen),
                        '测试前提：这些字符不应出现在训练语料中')
        ids = fresh.encode(unseen)
        self.assertEqual(set(ids), {UNK_ID})
        self.assertEqual(fresh.vocab_size, before, '编码不得扩充词表')
        self.assertEqual(sum(fresh.oov_tokens.values()), len(unseen))

    def test_pad_batch_marks_only_real_tokens(self):
        ids, mask = pad_batch([[5, 6], [7]])
        self.assertEqual(ids.shape, (2, 2))
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])
        self.assertEqual(ids[1].tolist(), [7, PAD_ID])

    def test_encoding_never_truncates_beyond_max_length(self):
        ids, mask = pad_batch([[1, 2, 3, 4, 5]], max_length=3)
        self.assertEqual(ids.shape[1], 3)
        self.assertEqual(mask.sum(), 3)


class AutogradTests(unittest.TestCase):
    def build_graph(self, seed=3, rows=4, length=6, hidden=5, classes=3):
        rng = np.random.default_rng(seed)
        table = Tensor(rng.normal(0, .1, (7, hidden)), requires_grad=True, name='table')
        weight = Tensor(rng.normal(0, .1, (hidden, classes)), requires_grad=True, name='weight')
        bias = Tensor(np.zeros(classes), requires_grad=True, name='bias')
        ids = rng.integers(0, 7, size=(rows, length))
        mask = np.ones((rows, length), dtype=np.int64)
        mask[1, 3:] = 0
        mask[2, 4:] = 0
        labels = rng.integers(0, classes, size=rows)
        return table, weight, bias, ids, mask, labels

    def graph_loss(self, table, weight, bias, ids, mask, labels) -> Tensor:
        """构建计算图并返回标量损失张量（供反向使用）。"""
        embedded = embedding(table, ids)
        pooled = masked_mean(embedded, mask)
        logits = linear(pooled, weight, bias)
        return softmax_cross_entropy(logits, one_hot(labels, 3))

    def scalar_loss(self, table, weight, bias, ids, mask, labels):
        def compute():
            return float(self.graph_loss(table, weight, bias, ids, mask, labels).data.reshape(-1)[0])
        return compute

    def test_analytical_matches_numerical_gradients(self):
        table, weight, bias, ids, mask, labels = self.build_graph()
        for parameter in (table, weight, bias):
            parameter.zero_grad()
        backward(self.graph_loss(table, weight, bias, ids, mask, labels))
        numeric = numerical_gradients(self.scalar_loss(table, weight, bias, ids, mask, labels),
                                      [table, weight, bias], epsilon=1e-6)
        for parameter, value in zip((table, weight, bias), numeric):
            self.assertLess(relative_error(parameter.grad, value), 1e-4,
                            f'{parameter.name}解析梯度与数值梯度不一致')

    def test_backward_follows_topological_order_not_reverse(self):
        """反向顺序写错时中间梯度会是0；这里直接检查池化层梯度非零。"""
        table, weight, bias, ids, mask, labels = self.build_graph()
        embedded = embedding(table, ids)
        pooled = masked_mean(embedded, mask)
        logits = linear(pooled, weight, bias)
        loss = softmax_cross_entropy(logits, one_hot(labels, 3))
        backward(loss)
        self.assertGreater(float(np.abs(pooled.grad).sum()), 0.0)
        self.assertGreater(float(np.abs(embedded.grad).sum()), 0.0)
        self.assertGreater(float(np.abs(table.grad).sum()), 0.0)

    def test_padding_never_changes_masked_mean(self):
        rng = np.random.default_rng(11)
        table = Tensor(rng.normal(0, .5, (5, 4)), requires_grad=False)
        ids = np.array([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0]])
        narrow_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        wide_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        embedded = embedding(table, ids)
        first = masked_mean(embedded, narrow_mask).data
        second = masked_mean(embedded, wide_mask).data
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_embedding_rejects_out_of_range_index(self):
        table = Tensor(np.zeros((3, 2)), requires_grad=True)
        with self.assertRaisesRegex(ValueError, '越界'):
            embedding(table, np.array([[0, 3]]))


class ModelTests(FixtureMixin, unittest.TestCase):
    def test_same_seed_trains_identically(self):
        first = MeanPoolClassifier(self.tokenizer.vocab_size, len(self.labels), ModelConfig(epochs=2, seed=5))
        second = MeanPoolClassifier(self.tokenizer.vocab_size, len(self.labels), ModelConfig(epochs=2, seed=5))
        known = self.validation_labels != UNKNOWN_ID
        train_model(first, self.train_arrays, self.train_labels,
                    (self.validation_arrays[0][known], self.validation_arrays[1][known]),
                    self.validation_labels[known])
        train_model(second, self.train_arrays, self.train_labels,
                    (self.validation_arrays[0][known], self.validation_arrays[1][known]),
                    self.validation_labels[known])
        np.testing.assert_array_equal(first.embedding_table.data, second.embedding_table.data)

    def test_label_shift_destroys_learning_signal(self):
        known = self.validation_labels != UNKNOWN_ID
        validation = (self.validation_arrays[0][known], self.validation_arrays[1][known])
        aligned = MeanPoolClassifier(self.tokenizer.vocab_size, len(self.labels), ModelConfig(epochs=8, seed=5))
        history = train_model(aligned, self.train_arrays, self.train_labels, validation,
                              self.validation_labels[known])
        shifted = MeanPoolClassifier(self.tokenizer.vocab_size, len(self.labels),
                                     ModelConfig(epochs=8, seed=5, label_shift=1))
        shifted_history = train_model(shifted, self.train_arrays, self.train_labels, validation,
                                      self.validation_labels[known])
        self.assertLess(history.train_loss[-1], shifted_history.train_loss[-1] / 2,
                        '标签错位应显著抬高训练损失')

    def test_one_hot_rejects_missing_labels(self):
        with self.assertRaisesRegex(ValueError, '缺失值'):
            one_hot(np.array([1.0, np.nan]), 3)

    def test_macro_f1_counts_unpredicted_class_as_zero(self):
        truth = np.array([0, 0, 1, 1])
        prediction = np.array([0, 0, 1, 1])
        self.assertEqual(macro_f1(truth, prediction, 3), 2 / 3)

    def test_summary_metrics_uses_same_denominator_as_lesson01(self):
        costs = {'known_wrong': 3, 'known_reject': 1, 'unknown_accept': 5}
        metrics = summary_metrics(['invoice', 'payment', 'unknown'],
                                  ['unknown', 'payment', 'invoice'], ['invoice', 'payment'], costs)
        self.assertEqual(metrics['known_n'], 2)
        self.assertEqual(metrics['known_coverage'], 0.5)
        self.assertEqual(metrics['unknown_false_accept_rate'], 1.0)
        self.assertEqual(metrics['routing_cost_per_request'], 2.0)


class DiagnosticsTests(FixtureMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MeanPoolClassifier(cls.tokenizer.vocab_size, len(cls.labels),
                                       ModelConfig(epochs=3, seed=9))

    def test_padding_in_mean_changes_representation_but_masked_mean_does_not(self):
        sensitivity = padding_sensitivity(self.model, self.test_arrays[0], self.test_arrays[1],
                                          self.test_labels, 128)
        self.assertGreater(sensitivity['max_pooled_abs_difference'], 0.01)

    def test_batch_width_invariance_separates_correct_and_buggy(self):
        report = batch_width_invariance(self.model, self.test_arrays[0], self.test_arrays[1],
                                        self.test_labels)
        self.assertLess(report['masked_mean']['max_pooled_drift'], 1e-12)
        self.assertGreater(report['padding_in_mean']['max_pooled_drift'], 0.05)

    def test_gradient_check_reports_small_relative_error(self):
        model = MeanPoolClassifier(self.tokenizer.vocab_size, len(self.labels), ModelConfig(seed=9))
        report = gradient_check(model, self.train_arrays[0], self.train_arrays[1],
                                self.train_labels, n_rows=4)
        for name in ['embedding_table', 'classifier_weight', 'classifier_bias']:
            self.assertLess(report[name]['max_relative_error'], 1e-4, name)


class DataContractTests(FixtureMixin, unittest.TestCase):
    def test_frozen_dataset_hash_and_counts(self):
        self.assertEqual(self.audit['rows'], 1280)
        self.assertEqual(self.audit['known_test_rows'], 240)
        self.assertEqual(self.audit['unknown_test_rows'], 40)
        self.assertEqual(self.audit['template_groups'], 256)
        for label in self.labels:
            self.assertEqual(self.audit['test_label_counts'][label], 20)

    def test_reference_predictions_align_with_test_split(self):
        baseline = load_frozen_baseline()
        baseline = baseline[baseline.split == 'test'].reset_index(drop=True)
        np.testing.assert_array_equal(baseline.sample_id.to_numpy(), self.test.sample_id.to_numpy())

    def test_challenges_are_separate_from_main_test(self):
        challenges = load_challenges()
        self.assertEqual(len(challenges), 24)
        self.assertTrue(set(challenges.text).isdisjoint(set(self.frame.text)))

    def test_tampered_frozen_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = ROOT.parent / 'lesson-01' / 'data'
            for name in ['samples.csv', 'manifest.json', 'templates.json', 'unknown_templates.json']:
                shutil.copy(source / name, Path(temp) / name)
            with (Path(temp) / 'samples.csv').open('a', encoding='utf-8') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'hash不匹配'):
                load_frozen(temp)


class ReferenceTests(FixtureMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = json.loads((REFERENCE / 'results.json').read_text(encoding='utf-8'))
        cls.predictions = pd.read_csv(REFERENCE / 'predictions.csv', keep_default_na=False)

    def test_reference_metrics_match_saved_predictions(self):
        known = self.predictions[self.predictions.intent != UNKNOWN]
        for column, name in [('mini_index_forced', 'mini_mean_pool_forced'),
                             ('mini_index_selected', 'mini_mean_pool')]:
            truth = known.intent_index.to_numpy()
            prediction = known[column].to_numpy()
            self.assertAlmostEqual(macro_f1(truth, prediction, len(self.labels)),
                                   self.results['metrics'][name]['known_macro_f1'], places=12)

    def test_reference_metrics_match_lesson01_baseline_exactly(self):
        lesson01 = json.loads((ROOT.parent / 'lesson-01' / 'reference' / 'baseline-v1' /
                               'results.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(self.results['metrics']['tfidf_frozen']['known_macro_f1'],
                               lesson01['metrics']['tfidf']['known_macro_f1'], places=12)
        self.assertAlmostEqual(self.results['metrics']['tfidf_forced_frozen']['known_macro_f1'],
                               lesson01['metrics']['tfidf_forced']['known_macro_f1'], places=12)

    def test_reference_uses_same_dataset_hash_as_lesson01(self):
        lesson01 = json.loads((ROOT.parent / 'lesson-01' / 'data' / 'manifest.json')
                              .read_text(encoding='utf-8'))
        self.assertEqual(self.results['dataset_sha256'], lesson01['sha256'])

    def test_reference_output_is_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, '输出目录非空'):
            run(REFERENCE)


if __name__ == '__main__':
    unittest.main()
