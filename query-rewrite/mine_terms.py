import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

PROJECT = Path('/home/ps/wuzhichao/agent_rag')
PARSED_DIR = PROJECT / 'data' / 'parsed'
MANIFEST = PROJECT / 'data' / 'manifest.jsonl'
OUT_DIR = PROJECT / 'query-rewrite' / 'glossary'
NL = chr(10)

RE_CHAPTER = re.compile(r'^第[一二三四五六七八九十百零0-9]+章\s+(.{1,24})$')
RE_ABBR = re.compile(
    r'([一-龥][一-龥（）()、]{1,29}?)'
    r'[（(]\s*以下(?:简称|统称)\s*[“\"]?\s*([^）)”"]{2,20}?)\s*[”\"]?\s*[)）]'
)
RE_DEF = re.compile(
    r'本(?:办法|细则|规定|制度|条例)所称(.{2,20}?)[，,、\s]*是指(.{5,160}?[。；])'
)
RE_DEF2 = re.compile(
    r'第[一二三四五六七八九十百零0-9]+条\s*([一-龥“”]{2,18}?)是指(.{5,160}?[。；])'
)
BOUNDARY = set('，。；：、！？“”‘’（）()《》【】[] \t　'
               '为对和与的把将在是指不也都又而但及或等被让向从由于并按据经因予')


def snap_left(text):
    cut = 0
    for i, ch in enumerate(text[:-1]):
        if ch in BOUNDARY:
            cut = i + 1
    return text[cut:]


RE_DEPT = re.compile(
    r'([一-龥]{2,12}?(?:管理部门|工作部|办公室|委员会))(?:是|为)[^。；]{0,24}归口'
)
RE_WORDS = {
    '假期类': re.compile(r'[一-龥]{1,8}假'),
    '费用类': re.compile(r'[一-龥]{2,8}费'),
    '补贴津贴类': re.compile(r'[一-龥]{2,10}(?:补贴|津贴)'),
    '保险福利类': re.compile(r'[一-龥]{2,8}(?:保险|公积金|年金)'),
    '人群主体类': re.compile(r'[一-龥]{2,10}(?:员工|人员|职工)'),
    '部门组织类': re.compile(r'[一-龥]{2,10}(?:管理部门|工作部|办公室|委员会)'),
}
RE_FORM = re.compile(r'《([^》]{2,30})》')

STOP_WORDS = set('''
请假 休假 放假 假期 假日 节假 节假日 请假条 准假 销假 假满 超假 旷假 病假条
事假条 年假条 假期内 休假期间 请假手续 请假者 批假 度假 寒假 暑假 假勤
费用 经费 资费 稿费 饭费 杂费 欠费 缴费 交费 付费 收费 免费 罚款 税费
关税 保费 运费 杂费 公费 自费 消费 浪费 旅费 会费 党费 团费
补贴 津贴
弄虚作假 其他费 成本费 相关费 浪费 申请经费 人均费 高消费
招待活动实行招待费 采取发放公务交通补贴 各类补贴
联网技术有限公司取暖补贴 联网技术有限公司住房补贴
停发骨干人才岗位津贴 公司骨干人才岗位津贴
依法参加基本养老保险 集团有限公司企业年金 技术有限公司企业年金
职工放弃参加企业年金 职工参加企业年金 通知本企业年金 职工企业年金
法定节假 国家法定节假
增加产假 会前请假 员工请假 请病假 计入年休假 员工请事假 其他假
报人力资源管理部门 交科研生产管理部门 工作部
岗职工 业人员 会人员 办人员 发展党员工 职职工
天物联网技术有限公司职工 天物联网技术有限公司员工
鼓励员工 通知参会人员 维护职工 追究相关人员 保障员工
身边工作人员 各级各类人员 其他无关人员 部门人员 部门员工
定人员 规范员工 禁止任何员工 其中职工 定有关人员 有关领导人员
技术有限公司返聘退休人员
'''.split())

STOP_PEOPLE = set('''
工作人员 管理人员 负责人员 相关人员 有关人员 其他人员 各类人员 人员情况
人员名单 专职人员 兼职人员 值班人员 当班人员 在岗人员 离职人员 退休人员
新进人员 外来人员 外部人员 来访人员 施工人员 作业人员 操作人员 使用人员
参加人员 与会人员 参会人员 培训人员 审计人员 检查人员 监督人员 审核人员
审批人员 经办人员 业务人员 技术人员 科研人员 保安人员 医务人员 勤杂人员
员工本人 员工群众 新员工 老员工 全体员工 广大员工
委员会 管理部门
'''.split())


def load_docs():
    docs = []
    for line in MANIFEST.open(encoding='utf-8'):
        meta = json.loads(line)
        if meta['status'] not in ('ok', 'empty'):
            continue
        path = PARSED_DIR / meta['parsed_file']
        body = []
        if path.exists():
            text = path.read_text(encoding='utf-8')
            if '# body' in text:
                body = [l for l in text.split('# body', 1)[1].split(NL) if l]
        meta['body'] = body
        docs.append(meta)
    return docs


class Agg:
    def __init__(self):
        self.count = 0
        self.docs = set()
        self.example = ''

    def add(self, doc_key, category, example):
        self.count += 1
        self.docs.add(doc_key)
        if not self.example and example:
            self.example = example[:100]


def example_line(body, frag, max_len=90):
    for line in body:
        if frag in line:
            i = line.find(frag)
            s = max(0, i - 20)
            seg = line[s:i + max_len]
            return ('...' if s else '') + seg
    return ''


def mine(docs):
    chapters = defaultdict(Agg)
    abbrs = defaultdict(Agg)
    defs = defaultdict(Agg)
    depts = defaultdict(Agg)
    words = {k: defaultdict(Agg) for k in RE_WORDS}
    forms = defaultdict(Agg)

    for d in tqdm(docs, desc='挖掘术语'):
        key = d['doc_title']
        cat = d['category']
        body = d['body']
        for line in body:
            m = RE_CHAPTER.match(line)
            if m and len(line) <= 26:
                ch = re.sub(r'\s+', '', m.group(1))
                chapters[ch].add(key, cat, line)

            for m in RE_ABBR.finditer(line):
                full = snap_left(m.group(1).strip('（）()、，。 　'))
                anchor = '航天物联网技术有限公司'
                if anchor in full:
                    full = anchor + full.split(anchor, 1)[1]
                short = m.group(2).strip('“”" ')
                if 3 <= len(full) <= 30 and 2 <= len(short) <= 16:
                    abbrs[(full, short)].add(key, cat, line)

            for rx in (RE_DEF, RE_DEF2):
                for m in rx.finditer(line):
                    term = m.group(1).strip('“”" 　')
                    term = re.sub(
                        r'^本(?:办法|细则|规定|制度|章程)?(?:所称|所指|中的?|管理的)',
                        '', term,
                    ).strip('“”" 　')
                    definition = m.group(2).strip()
                    if 2 <= len(term) <= 20:
                        defs[term].add(key, cat, term + '：' + definition)

            m = RE_DEPT.search(line)
            if m:
                depts[snap_left(m.group(1))].add(key, cat, line)

            for kind, rx in RE_WORDS.items():
                for raw_w in set(rx.findall(line)):
                    w = snap_left(raw_w)
                    if w in STOP_WORDS or w in STOP_PEOPLE or len(w) < 2:
                        continue
                    words[kind][w].add(key, cat, example_line(body, w))

            for f in set(RE_FORM.findall(line)):
                forms[f].add(key, cat, line)

    return chapters, abbrs, defs, depts, words, forms


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def agg_rows(agg):
    out = []
    for k, a in agg.items():
        out.append((k, a.count, len(a.docs), sorted(a.docs), a.example))
    out.sort(key=lambda x: (-x[2], -x[1], str(x[0])))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    chapters, abbrs, defs, depts, words, forms = mine(docs)

    write_csv(
        OUT_DIR / '02_章标题词表.csv',
        ['章标题', '出现次数', '覆盖制度数', '来源制度', '原句'],
        [(k, c, nd, '；'.join(ds), ex) for k, c, nd, ds, ex in agg_rows(chapters)],
    )
    abbr_rows = [
        (f, s, c, nd, '；'.join(ds))
        for (f, s), c, nd, ds, ex in agg_rows(abbrs)
        if c >= 2 or '有限公司' in f or '研究院' in f
    ]
    write_csv(
        OUT_DIR / '03_简称对照表.csv',
        ['全称', '简称', '出现次数', '覆盖制度数', '来源制度'],
        abbr_rows,
    )
    write_csv(
        OUT_DIR / '04_术语定义表.csv',
        ['术语', '出现次数', '覆盖制度数', '来源制度', '定义原句'],
        [(k, c, nd, '；'.join(ds), ex) for k, c, nd, ds, ex in agg_rows(defs)],
    )
    write_csv(
        OUT_DIR / '05_部门组织表.csv',
        ['部门/组织', '出现次数', '覆盖制度数', '来源制度', '原句'],
        [(k, c, nd, '；'.join(ds), ex) for k, c, nd, ds, ex in agg_rows(depts)],
    )
    min_count = {'假期类': 2, '人群主体类': 3}
    for kind, bucket in words.items():
        threshold = min_count.get(kind, 2)
        rows = [
            (k, a.count, len(a.docs), '；'.join(sorted(a.docs)), a.example)
            for k, a in sorted(bucket.items(), key=lambda kv: (-len(kv[1].docs), -kv[1].count))
            if a.count >= threshold
        ]
        write_csv(
            OUT_DIR / ('06_候选领域词_' + kind + '.csv'),
            ['候选词', '出现次数', '覆盖制度数', '来源制度', '例句'],
            rows,
        )
    write_csv(
        OUT_DIR / '07_表单附件名.csv',
        ['表单/附件名', '出现次数', '覆盖制度数', '来源制度'],
        [(k, c, nd, '；'.join(ds)) for k, c, nd, ds, ex in agg_rows(forms)],
    )

    title_docs = defaultdict(list)
    for d in docs:
        title_docs[d['doc_title']].append(d)

    list_rows = []
    for d in sorted(docs, key=lambda x: (x['category'], x['doc_title'])):
        versions = len(title_docs[d['doc_title']])
        list_rows.append([
            d['category'], d['doc_title'], d['doc_number'], d['ext'],
            d['chars'], '多版本' if versions > 1 else '',
            d['source'], d['status'],
        ])
    write_csv(
        OUT_DIR / '00_制度清单.csv',
        ['大类', '制度名称', '文号', '格式', '正文字数', '版本标记', '源路径', '解析状态'],
        list_rows,
    )

    multi = {t: ds for t, ds in title_docs.items() if len(ds) > 1}
    md = ['# 制度文档清单（130 份，13 个大类）', NL]
    cats = defaultdict(list)
    for d in docs:
        cats[d['category']].append(d)
    for cat in sorted(cats):
        md.append('## ' + cat + '（' + str(len(cats[cat])) + '份）')
        for d in sorted(cats[cat], key=lambda x: x['doc_title']):
            num = '　' + d['doc_number'] if d['doc_number'] else ''
            flag = ' ⚠多版本' if len(title_docs[d['doc_title']]) > 1 else ''
            md.append('- ' + d['doc_title'] + num + flag)
        md.append('')
    (OUT_DIR / '00_制度清单.md').write_text(NL.join(md), encoding='utf-8')

    print('文档数:', len(docs))
    print('章标题:', len(chapters), ' 简称对:', len(abbrs),
          ' 术语定义:', len(defs), ' 归口部门:', len(depts), ' 表单名:', len(forms))
    print('同名多版本制度:', len(multi), '个 ->', '；'.join(sorted(multi)))
    for kind, bucket in words.items():
        kept = sum(1 for a in bucket.values() if a.count >= min_count.get(kind, 2))
        print(kind, '候选词:', kept)


if __name__ == '__main__':
    main()
