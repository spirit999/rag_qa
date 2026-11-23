import json
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

PROJECT = Path('/home/ps/wuzhichao/agent_rag')
DOCS_DIR = PROJECT / 'docs'
CONV_DIR = PROJECT / 'data' / 'converted'
PARSED_DIR = PROJECT / 'data' / 'parsed'
MANIFEST = PROJECT / 'data' / 'manifest.jsonl'

W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NL = chr(10)


def parse_docx(path):
    blocks = []
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read('word/document.xml')
    root = ET.fromstring(xml_bytes)
    body = root.find(W_NS + 'body')
    for node in list(body):
        tag = node.tag
        if tag == W_NS + 'p':
            text = ''.join(t.text or '' for t in node.iter(W_NS + 't'))
            text = re.sub(r'[ \t ]+', ' ', text).strip()
            if text:
                blocks.append(('p', text))
        elif tag == W_NS + 'tbl':
            rows = []
            for tr in node.findall(W_NS + 'tr'):
                cells = []
                for tc in tr.findall(W_NS + 'tc'):
                    cell_text = ''.join(t.text or '' for t in tc.iter(W_NS + 't')).strip()
                    cells.append(re.sub(r'\s+', ' ', cell_text))
                row = ' | '.join(c for c in cells if c)
                if row:
                    rows.append(row)
            if rows:
                blocks.append(('tbl', NL.join(rows)))
    return blocks


PARA_START = re.compile(
    r'^(第[一二三四五六七八九十百零0-9]+[章节条]'
    r'|[（(][一二三四五六七八九十百0-9]+[)）]'
    r'|\d+[.、]'
    r'|【)'
)
CJK = re.compile(r'[一-鿿]')


def join_wrap(prev, cur):
    if prev and CJK.search(prev[-1]) and cur and CJK.search(cur[0]):
        return prev + cur
    return prev + ' ' + cur


def parse_pdf(path):
    out = subprocess.run(
        ['pdftotext', '-layout', str(path), '-'],
        capture_output=True, text=True, timeout=120,
    )
    paragraphs = []
    cur = ''
    for raw in out.stdout.split(NL):
        line = re.sub(r'[ \t ]+', ' ', raw).strip()
        if not line:
            if cur:
                paragraphs.append(cur)
                cur = ''
            continue
        if re.fullmatch(r'\d{1,3}', line):
            continue
        if PARA_START.match(line):
            if cur:
                paragraphs.append(cur)
            cur = line
        elif cur:
            cur = join_wrap(cur, line)
        else:
            cur = line
    if cur:
        paragraphs.append(cur)
    return [('p', p) for p in paragraphs]


def parse_xlsx(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True, read_only=True)
    blocks = []
    for ws in wb.worksheets:
        blocks.append(('p', '【工作表：' + ws.title + '】'))
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                blocks.append(('p', ' | '.join(cells)))
    wb.close()
    return blocks


def parse_filename_meta(path):
    name = path.stem
    name = name.replace('（公开）', '').replace('(公开)', '').strip()
    doc_number = ''
    title = name
    parts = re.split(r'[\s\-—]+', name)
    if len(parts) > 1 and re.search(r'[〔\[]\s*\d{4}\s*[〕\]]', parts[-1]):
        doc_number = parts[-1].strip()
        title = ''.join(parts[:-1]).strip()
    title = re.sub(r'^航天物联网技术有限公司', '', title).strip()
    return title, doc_number


def convert_doc_group(files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'soffice', '--headless',
        '-env:UserInstallation=file:///tmp/lo_agent_rag',
        '--convert-to', 'docx',
        '--outdir', str(out_dir),
    ] + [str(f) for f in files]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return res.returncode == 0, res.stdout + res.stderr


def main():
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    all_files = sorted(p for p in DOCS_DIR.rglob('*') if p.is_file())
    by_ext = defaultdict(list)
    for p in all_files:
        by_ext[p.suffix.lower()].append(p)

    doc_by_cat = defaultdict(list)
    for p in by_ext.get('.doc', []):
        doc_by_cat[p.parent.name].append(p)
    for cat, files in tqdm(doc_by_cat.items(), desc='libreoffice转换doc'):
        out_dir = CONV_DIR / cat
        pending = [f for f in files if not (out_dir / (f.stem + '.docx')).exists()]
        if pending:
            ok, log = convert_doc_group(pending, out_dir)
            if not ok:
                print('转换失败:', cat, log[:500])

    records = []
    for path in tqdm(all_files, desc='解析文档'):
        rel = path.relative_to(DOCS_DIR)
        cat = rel.parts[0]
        ext = path.suffix.lower()
        title, doc_number = parse_filename_meta(path)
        blocks, status, note = [], 'ok', ''
        try:
            if ext == '.docx':
                blocks = parse_docx(path)
            elif ext == '.doc':
                converted = CONV_DIR / cat / (path.stem + '.docx')
                if converted.exists():
                    blocks = parse_docx(converted)
                else:
                    status, note = 'error', '转换后文件不存在'
            elif ext == '.pdf':
                blocks = parse_pdf(path)
            elif ext in ('.xlsx', '.xls'):
                blocks = parse_xlsx(path)
            else:
                status, note = 'skip', '不支持的类型'
        except Exception as e:
            status, note = 'error', repr(e)[:200]

        safe_stem = re.sub(r'[/\\:*?"<>|]', '_', path.stem)
        out_name = cat + '__' + safe_stem + '.txt'
        out_path = PARSED_DIR / out_name
        lines = ['# meta']
        lines.append('category: ' + cat)
        lines.append('source: ' + str(rel))
        lines.append('doc_title: ' + title)
        lines.append('doc_number: ' + doc_number)
        lines.append('n_tables: ' + str(sum(1 for k, _ in blocks if k == 'tbl')))
        lines.append('# body')
        for kind, text in blocks:
            if kind == 'tbl':
                lines.append('[表格]')
                lines.append(text)
            else:
                lines.append(text)
        out_path.write_text(NL.join(lines), encoding='utf-8')

        n_para = sum(1 for k, _ in blocks if k == 'p')
        chars = sum(len(t) for _, t in blocks)
        if chars == 0:
            status = 'empty'
        records.append({
            'category': cat,
            'source': str(rel),
            'ext': ext,
            'doc_title': title,
            'doc_number': doc_number,
            'parsed_file': out_name,
            'paragraphs': n_para,
            'tables': sum(1 for k, _ in blocks if k == 'tbl'),
            'chars': chars,
            'status': status,
            'note': note,
        })

    with MANIFEST.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + NL)

    print(NL + '=== 解析汇总 ===')
    stat = defaultdict(int)
    for r in records:
        stat[(r['ext'], r['status'])] += 1
    for k in sorted(stat):
        print(k, stat[k])
    print('总文档数:', len(records))
    print('输出目录:', PARSED_DIR)


if __name__ == '__main__':
    main()
