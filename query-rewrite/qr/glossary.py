import csv
from collections import defaultdict

from .config import CATEGORIES, GLOSSARY_DIR

CORE_CSV = GLOSSARY_DIR / '01_核心术语对照表.csv'
DOC_LIST_CSV = GLOSSARY_DIR / '00_制度清单.csv'


def title_digest(path=DOC_LIST_CSV, max_chars=2400):
    """按大类分组的制度全称清单（去重，多版本只保留一个），注入 prompt 用。"""
    by_cat = defaultdict(list)
    seen = set()
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            t = r['制度名称'].strip()
            if not t or t in seen:
                continue
            seen.add(t)
            by_cat[r['大类'].strip()].append(t)
    lines = []
    total = 0
    for cat in CATEGORIES:
        if cat not in by_cat:
            continue
        line = cat + '：' + '、'.join(by_cat[cat])
        if total + len(line) > max_chars:
            lines.append('……（其余制度略）')
            break
        lines.append(line)
        total += len(line)
    return chr(10).join(lines)


class Glossary:
    def __init__(self, path=CORE_CSV):
        self.rows = []
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                term = r['公文术语（规范说法）'].strip()
                colloq = [s for s in r['口语/同义说法（建议待人工复核）'].split('/') if s.strip()]
                self.rows.append({
                    'category': r['类别'].strip(),
                    'term': term,
                    'colloquial': [s.strip() for s in colloq],
                    'source': r['来源制度/依据'].strip(),
                    'note': r['备注'].strip(),
                })
        self.synonym_map = {}
        for row in self.rows:
            for c in row['colloquial']:
                self.synonym_map.setdefault(c, set()).add(row['term'])
        self.all_surface = sorted(
            set(self.synonym_map) | {row['term'] for row in self.rows},
            key=len, reverse=True,
        )

    def prompt_digest(self, max_chars=2200):
        groups = defaultdict(list)
        for row in self.rows:
            alias = ('（' + '/'.join(row['colloquial']) + '）') if row['colloquial'] else ''
            groups[row['category']].append(row['term'] + alias)
        lines = []
        total = 0
        for cat in groups:
            line = cat + '：' + '；'.join(groups[cat])
            if total + len(line) > max_chars:
                lines.append('……（其余术语略）')
                break
            lines.append(line)
            total += len(line)
        return chr(10).join(lines)

    def match_terms(self, text):
        hits = set()
        for surface in self.all_surface:
            if surface in text:
                if surface in self.synonym_map:
                    hits.update(self.synonym_map[surface])
                hits.add(surface)
        return sorted(hits)
