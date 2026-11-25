"""Check P00 deliverable structure and recorded witness invariants."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'evidence/P00'
TOP = ['README', 'MASTER_DESIGN', 'REQUIREMENTS', 'CONTRACTS', 'REPOSITORY_AUDIT', 'ROADMAP',
       'TRACEABILITY_MATRIX', 'PROJECT_STATE', 'DECISION_LOG', 'RISK_REGISTER', 'OPEN_ISSUES', 'EVALUATION_PROTOCOL']


def read(path):
    return path.read_text(encoding='utf-8')


def evidence(name):
    doc = json.loads(read(EVIDENCE/f'{name}.json'))
    assert doc['section'] == name and doc['observed']
    return doc['result']


def rows(path):
    entries = [line.strip('|').split('|') for line in read(path).splitlines()
               if line.startswith('| S05-REQ-')]
    return [[value.strip() for value in row] for row in entries]


def anchors(path):
    titles = re.findall(r'^#{1,6} (.+)$', read(path), flags=re.M)
    return {re.sub(r'[^\w -]', '', title.lower()).replace(' ', '-') for title in titles}


def links(files):
    total = 0
    for src in files:
        for match in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', read(src)):
            if match.startswith(('https://', 'http://')):
                continue
            name, _, anchor = match.partition('#')
            target = (src.parent/name).resolve() if name else src
            assert target.is_relative_to(ROOT), (src, match)
            assert target.exists(), (src, match)
            if anchor and target.suffix == '.md':
                assert anchor in anchors(target), (src, match, sorted(anchors(target)))
            total += 1
    return total


def main():
    files = list(ROOT.rglob('*.md'))
    for name in TOP:
        assert (ROOT/f'{name}.md').is_file(), name
    assert (ROOT/'reports/P00_acceptance.md').is_file()
    req = rows(ROOT/'REQUIREMENTS.md')
    matrix = rows(ROOT/'TRACEABILITY_MATRIX.md')
    ids = [f'S05-REQ-{i:03d}' for i in range(1, 31)]
    assert [r[0] for r in req] == [r[0] for r in matrix] == ids
    assert all(len(row) == 9 and all(row) for row in matrix)
    assert [r[2] for r in req] == [r[3] for r in matrix]
    phases = sorted((ROOT/'specs').glob('P??_*.md'))
    assert [p.name[:3] for p in phases] == [f'P{i:02d}' for i in range(11)]
    for spec in phases:
        doc = read(spec)
        sections = [int(x) for x in re.findall(r'^## (\d+)\. ', doc, re.M)]
        assert sections == list(range(1,26)), spec
        cover = doc.split('## 4. ',1)[1].split('## 5. ',1)[0]
        for req_id, _, owner, _ in req:
            if owner == spec.name[:3]:
                assert req_id[-3:] in cover, (spec, req_id)
    local_links = links(files)
    raw, data, host, s04 = (evidence(i) for i in ('raw','data','host','s04'))
    follow, existing, source, code = (evidence(i) for i in ('followup','tests','source','code'))
    assert len(host['git']) == 4 and all(v['commands'][0]['exit'] == 0 for v in host['git'])
    assert data['paired_episode_count'] == raw['resolved_episode_slots'] == 30
    assert all(x['inspect']['exit'] == 0 for x in raw['inspected_bags'])
    assert raw['model']['probe']['exit'] == 0 and '54 54 54 54' in raw['model']['probe']['stdout']
    assert s04['provenance']['action_dim'] == 14 and s04['provenance']['command_dim'] == 54
    assert s04['reviews'] and all(r['actor_sha256'] for r in s04['reviews'])
    assert existing['exit'] == 2
    assert follow['bag_witness']['exit'] == 0 and 'tracker_right' in follow['bag_witness']['stdout']
    assert follow['wrapped_tests']['exit'] == 0 and '6 passed' in follow['wrapped_tests']['stdout']
    assert len(source) > 25 and len(code) > 25
    gate_rows = [l for l in read(ROOT/'reports/P00_acceptance.md').splitlines()
                 if re.match(r'^\| G\d{2}\b', l)]
    assert len(gate_rows) == 17 and all('| PASS |' in l for l in gate_rows)
    state = read(ROOT/'PROJECT_STATE.md')
    assert 'P00 | READY_FOR_REVIEW' in state and 'P01–P10 | NOT_STARTED' in state
    git = subprocess.run(['git','diff','--binary','HEAD'], cwd=ROOT, capture_output=True, check=True)
    print(f'PASS {len(files)} Markdown documents, {len(ids)} requirements, {len(phases)} phases with 25 sections each')
    print(f'PASS {local_links} relative Markdown links, 17 acceptance rows and 8 machine-readable witness files')
    print('tracked_diff_sha256',hashlib.sha256(git.stdout).hexdigest())


if __name__ == '__main__':
    main()
