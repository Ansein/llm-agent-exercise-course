"""Deterministic synthetic fixture generation. Never overwrite frozen data by default."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ["{text}。", "你好，{text}？", "想请你帮忙：{text}。", "我这边的情况是：{text}，谢谢。", "麻烦确认一下，{text}。"]
FIELDS = ['sample_id', 'customer_id', 'template_id', 'text', 'intent', 'split', 'source', 'dataset_version']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_rows():
    templates = json.loads((ROOT/'data/templates.json').read_text())
    unknown = json.loads((ROOT/'data/unknown_templates.json').read_text())
    config = json.loads((ROOT/'config.json').read_text())
    rng = random.Random(config['seed'])
    rows = []
    # Customer IDs carry no business label; one customer owns one phrase family.
    customers = list(range(1, 257))
    rng.shuffle(customers)
    groups = 0
    for intent, phrases in templates.items():
        if len(phrases) != 20:
            raise ValueError(f'{intent}: expected 20 authored phrase families')
        for i, phrase in enumerate(phrases):
            split = 'train' if i < 12 else 'validation' if i < 16 else 'test'
            for wrapper in WRAPPERS:
                rows.append(dict(customer_id=f'C{customers[groups]:04}', template_id=f'{intent}-{i:02}',
                    text=wrapper.format(text=phrase), intent=intent, split=split,
                    source='course-authored-synthetic', dataset_version=config['dataset_version']))
            groups += 1
    for split, phrases in unknown.items():
        for i, phrase in enumerate(phrases):
            for wrapper in WRAPPERS:
                rows.append(dict(customer_id=f'C{customers[groups]:04}', template_id=f'unknown-{split}-{i:02}',
                    text=wrapper.format(text=phrase), intent='unknown', split=split,
                    source='course-authored-synthetic', dataset_version=config['dataset_version']))
            groups += 1
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row['sample_id'] = f'S{i+1:04}'
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'samples.csv'
    if path.exists():
        sys.exit('Refusing to overwrite frozen samples.csv. Use a new --output-dir for comparison.')
    rows = build_rows()
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        'dataset_version': '1.0.0', 'seed': 20260914,
        'description': '1200 known-intent samples plus 80 OOD samples; synthetic only',
        'sha256': sha256(path), 'rows': len(rows),
        'known_rows': sum(r['intent'] != 'unknown' for r in rows),
        'counts': {s: sum(r['split'] == s for r in rows) for s in ['train','validation','test']},
        'template_groups': groups_count(rows, 'template_id'),
        'customer_groups': groups_count(rows, 'customer_id'),
        'source_hashes': {name: sha256(ROOT/'data'/name) for name in ['templates.json','unknown_templates.json']},
        'split_policy': 'first 12/next 4/last 4 authored families per known label; OOD 8 validation and 8 test families; five variants per family',
        'limitation': 'Customer and template groups coincide in this fixture. Semantic and stylistic overlap remains; no production-generalization claim.'
    }
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

def groups_count(rows, key):
    return len({r[key] for r in rows})

if __name__ == '__main__':
    main()
