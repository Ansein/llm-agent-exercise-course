"""Text-only baselines. Training sees neither IDs nor split/template metadata."""
import json
import unicodedata
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = 'unknown'


def normalize(text):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    return ''.join(unicodedata.normalize('NFKC', text).lower().split())


class KeywordRules:
    """One matching intent -> route; zero/multiple intents -> human review.

    Deliberately simple teaching baseline: substring matches do not understand negation.
    """
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else json.loads((ROOT/'data/rules.json').read_text())

    def explain(self, text):
        text = normalize(text)
        return {label: [term for term in terms if term in text]
                for label, terms in self.rules.items() if any(term in text for term in terms)}

    def predict(self, texts):
        predictions = []
        for text in texts:
            matches = self.explain(text)
            predictions.append(next(iter(matches)) if len(matches) == 1 else UNKNOWN)
        return np.asarray(predictions)


def make_model(config):
    params = dict(config['tfidf'])
    params['ngram_range'] = tuple(params['ngram_range'])
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=normalize, **params)),
        ('classifier', LogisticRegression(**config['logistic']))
    ])


def scored_predict(model, texts):
    probabilities = model.predict_proba(texts)
    winners = probabilities.argmax(axis=1)
    return model.classes_[winners], probabilities.max(axis=1)


def apply_threshold(labels, scores, threshold):
    """These are uncalibrated model scores, not guaranteed probabilities of correctness."""
    return np.where(np.asarray(scores) >= threshold, labels, UNKNOWN)


def routing_cost(truth, predictions, costs):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    known = truth != UNKNOWN
    rejects = predictions == UNKNOWN
    return (np.sum(known & rejects)*costs['known_reject']
            + np.sum(known & ~rejects & (truth != predictions))*costs['known_wrong']
            + np.sum(~known & ~rejects)*costs['unknown_accept']) / len(truth)


def choose_threshold(truth, labels, scores, config):
    trials = []
    for threshold in config['threshold_grid']:
        pred = apply_threshold(labels, scores, threshold)
        known = np.asarray(truth) != UNKNOWN
        trials.append(dict(threshold=threshold,
            cost=float(routing_cost(truth, pred, config['costs'])),
            known_coverage=float(np.mean(pred[known] != UNKNOWN)),
            unknown_false_accept_rate=float(np.mean(pred[~known] != UNKNOWN))))
    # Predeclared tie-breaking rule: lower cost, then fewer OOD accepts, then more coverage.
    best = min(trials, key=lambda x: (x['cost'], x['unknown_false_accept_rate'], -x['known_coverage'], x['threshold']))
    return best['threshold'], trials
