"""Meaningful regression checks for leakage, metric semantics and frozen fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluate import load_data, validate_frame, summary_metrics, run, LABEL_NAMES
from baselines import KeywordRules, make_model, choose_threshold, apply_threshold
from generate_data import build_rows


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df,cls.audit=load_data()
        cls.config=json.loads((ROOT/'config.json').read_text())

    def test_frozen_counts_and_balanced_test(self):
        known=self.df[self.df.intent!='unknown']
        self.assertEqual(len(known),1200)
        self.assertEqual(known[known.split=='test'].groupby('intent').size().tolist(),[20]*12)
        self.assertEqual(self.audit['template_groups'],256)

    def test_customer_leak_is_rejected(self):
        df=self.df.copy()
        train=df.index[df.split=='train'][0];test=df.index[df.split=='test'][0]
        df.loc[test,'customer_id']=df.loc[train,'customer_id']
        with self.assertRaisesRegex(ValueError,'customer_id leakage'):validate_frame(df)

    def test_family_leak_is_rejected(self):
        df=self.df.copy()
        train=df.index[df.split=='train'][0];test=df.index[df.split=='test'][0]
        df.loc[test,'template_id']=df.loc[train,'template_id']
        with self.assertRaisesRegex(ValueError,'template_id leakage'):validate_frame(df)

    def test_normalized_duplicate_is_rejected(self):
        df=self.df.copy()
        df.loc[df.index[1],'text']=' '+df.loc[df.index[0],'text']+'！！！'
        with self.assertRaisesRegex(ValueError,'Duplicate normalized'):validate_frame(df)

    def test_frozen_bytes(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            for name in ['samples.csv','manifest.json','templates.json','unknown_templates.json']:
                shutil.copy(ROOT/'data'/name,Path(temp)/name)
            with (Path(temp)/'samples.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_data(temp)

    def test_generator_matches_all_rows(self):
        actual=self.df.to_dict('records')
        generated=build_rows()
        self.assertEqual(actual,generated)

    def test_unseen_validation_feature_never_fitted(self):
        model=make_model(self.config)
        model.fit(['物流查件','快递查件','开具发票','发票开具'],['shipping_status']*2+['invoice']*2)
        before=dict(model.named_steps['tfidf'].vocabulary_)
        model.predict_proba(['火星独角兽从未见过'])
        self.assertEqual(before,model.named_steps['tfidf'].vocabulary_)
        self.assertNotIn('独角兽',before)

    def test_rejection_not_counted_as_known_success(self):
        metrics=summary_metrics(['invoice','payment','unknown'],['unknown','payment','invoice'],
                                ['invoice','payment'],self.config['costs'])
        self.assertEqual(metrics['known_accuracy'],.5)
        self.assertEqual(metrics['unknown_false_accept_rate'],1.)
        self.assertEqual(metrics['known_rejected_n'],1)
        self.assertEqual(metrics['routing_cost_per_request'],2.)  # (reject 1 + OOD accept 5)/3

    def test_threshold_validation_cost_not_accuracy(self):
        config={'threshold_grid':[.1,.5,.9],'costs':self.config['costs']}
        threshold,_=choose_threshold(['invoice','unknown'],['invoice','invoice'],[.8,.2],config)
        self.assertEqual(threshold,.5)
        self.assertEqual(apply_threshold(['invoice'],[.5],.5).tolist(),['invoice'])

    def test_rules_abstain_on_conflict_and_empty(self):
        rules=KeywordRules()
        self.assertEqual(rules.predict(['','取消订单并开票']).tolist(),['unknown','unknown'])
        self.assertEqual(rules.predict(['开具发票']).tolist(),['invoice'])

    def test_metrics_independently_from_saved_predictions(self):
        p=pd.read_csv(ROOT/'reference/baseline-v1/predictions.csv')
        r=json.loads((ROOT/'reference/baseline-v1/results.json').read_text())
        known=p[p.intent!='unknown']
        for model,col in [('rules','rule_prediction'),('tfidf','tfidf_prediction')]:
            f=[]
            for label in r['label_order']:
                tp=int(((known.intent==label)&(known[col]==label)).sum())
                fp=int(((known.intent!=label)&(known[col]==label)).sum())
                fn=int(((known.intent==label)&(known[col]!=label)).sum())
                f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            self.assertAlmostEqual(sum(f)/12,r['metrics'][model]['known_macro_f1'],places=12)
            cm=pd.read_csv(ROOT/f'reference/baseline-v1/confusion-{model}.csv',index_col=0)
            self.assertEqual(cm.to_numpy().sum(),280)

    def test_reference_is_not_overwritten(self):
        with self.assertRaisesRegex(ValueError,'not empty'):run(ROOT/'reference/baseline-v1')

if __name__=='__main__':unittest.main()
