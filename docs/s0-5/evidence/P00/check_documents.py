"""Validate documentation, recorded evidence integrity, and phase log consistency.

This checker does not decide whether a technical claim or approval is sound.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'evidence/P00_review_fix'
TOP = ('README', 'MASTER_DESIGN', 'REQUIREMENTS', 'CONTRACTS', 'REPOSITORY_AUDIT',
       'ROADMAP', 'TRACEABILITY_MATRIX', 'PROJECT_STATE', 'DECISION_LOG', 'RISK_REGISTER',
       'OPEN_ISSUES', 'EVALUATION_PROTOCOL', 'EXECUTION_BASELINE', 'HISTORICAL_DATA_USAGE')
PHASES = tuple(f'P{i:02}' for i in range(11))
HEX = re.compile(r'[0-9a-f]{64}')
STATUS = {'NOT_STARTED', 'IN_PROGRESS', 'BLOCKED', 'FAILED', 'READY_FOR_REVIEW', 'ACCEPTED'}
EDGES = {
    'NOT_STARTED': {'IN_PROGRESS'},
    'IN_PROGRESS': {'BLOCKED', 'FAILED', 'READY_FOR_REVIEW'},
    'BLOCKED': {'IN_PROGRESS'},
    'FAILED': {'IN_PROGRESS'},
    'READY_FOR_REVIEW': {'ACCEPTED', 'IN_PROGRESS', 'BLOCKED'},
    'ACCEPTED': set(),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return path.read_text(encoding='utf-8')


def load_json(path):
    return json.loads(read(path))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(root, relative):
    candidate = Path(relative)
    require(not candidate.is_absolute() and '..' not in candidate.parts, f'Invalid path {relative}')
    path = (root / candidate).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f'Missing or external file {relative}')
    return path


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True)


def historical_snapshot(require_git):
    manifest = load_json(REVIEW / 'p00_snapshot_manifest.json')
    require(manifest['schema'] == 's05.p00.historical-snapshot.v1', 'Snapshot schema')
    commit = manifest['commit']
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'Invalid historic commit')
    root = manifest['root']
    rows = manifest['files']
    paths = [r['path'] for r in rows]
    require(len(paths) == len(set(paths)) == 38 and paths == sorted(paths), 'Historic path list')
    for row in rows:
        require(HEX.fullmatch(row['sha256']) and row['size'] >= 0, 'Historic file descriptor')
    formatted = ''.join(f"{r['path']} {r['sha256']}\n" for r in rows).encode()
    require(digest(formatted) == manifest['tree_sha256'], 'Historic manifest digest')
    if git('cat-file', '-e', f'{commit}^{{commit}}').returncode:
        if require_git:
            raise ValueError(f'Historic commit object is missing: {commit}')
        print(f'BLOCKED historic Git replay on this checkout: {commit}')
        return
    tree = git('ls-tree', '-r', '-z', '--full-tree', commit, '--', root)
    require(tree.returncode == 0, 'Unable to enumerate historic tree')
    blob_by_path = {}
    for item in tree.stdout.split(b'\0'):
        if not item:
            continue
        metadata, path = item.split(b'\t', 1)
        mode, kind, sha = metadata.decode().split()
        require(mode == '100644' and kind == 'blob', 'Unexpected historic file mode')
        rel = path.decode().removeprefix(root+'/')
        blob_by_path[rel] = sha
    require(set(blob_by_path) == set(paths), 'Historic tree differs from manifest')
    for row in rows:
        sha = blob_by_path[row['path']]
        require(sha == row['git_blob_sha1'], f"Historic blob mismatch: {row['path']}")
        blob = git('cat-file', 'blob', sha)
        require(blob.returncode == 0 and len(blob.stdout) == row['size']
                and digest(blob.stdout) == row['sha256'], f"Historic bytes differ: {row['path']}")
    print(f'PASS historic Git replay of {len(paths)} files at {commit}')


def phase_states(document):
    require(document['schema'] == 's05.phase-state.v1', 'Invalid phase log schema')
    states = dict.fromkeys(PHASES, 'NOT_STARTED')
    prior_time = None
    for event in document['events']:
        phase, target = event['phase'], event['to']
        require(phase in states and target in STATUS, 'Unknown phase/status')
        now = datetime.fromisoformat(event['at'])
        require(now.tzinfo is not None and (prior_time is None or prior_time <= now), 'Phase event chronology')
        prior_time = now
        source = states[phase]
        require(event['from'] == source and target in EDGES[source], f'Illegal {phase} {source} -> {target}')
        require(bool(event.get('evidence_ref')), f'Missing evidence for {phase} {target}')
        if target == 'IN_PROGRESS':
            require(bool(event.get('authorization_ref')), f'Missing start/resume authorization for {phase}')
            index = PHASES.index(phase)
            if index:
                require(states[PHASES[index-1]] == 'ACCEPTED', f'Previous phase not accepted for {phase}')
                require(bool(HEX.fullmatch(event.get('spec_sha256',''))), f'Missing frozen spec for {phase}')
            if phase == 'P09':
                gate = event.get('final_test', {})
                require(gate.get('status') == 'FINAL_TEST_SEALED' and
                        bool(HEX.fullmatch(gate.get('manifest_sha256',''))) and bool(gate.get('access_log_ref')),
                        'P09 requires an independent sealed final test')
        if target == 'ACCEPTED':
            approval = event.get('approval', {})
            require(approval.get('type') == 'human' and bool(approval.get('by')) and bool(approval.get('ref')),
                    f'Human approval evidence required for {phase}')
        states[phase] = target
    require(document['current'] == states, 'Phase replay differs from current state')
    if 'dependency_dcr' in document:
        p01_dependency_states(document)
    return states


def p01_dependency_states(document):
    dcr = document['dependency_dcr']
    require(dcr.get('id') == 'DCR-001' and dcr.get('approval',{}).get('type')=='human',
            'P01 dependency change requires a human approval')
    require(dcr.get('mandatory_dependencies')=={
        'P01-C':['P01-B-HMB'],'P01 exit':['P01-F'],'P03':['P02','P01-T']},
        'DCR-001 dependency configuration drifted')
    approved=datetime.fromisoformat(dcr['approved_at'])
    checked_file(ROOT,dcr['approval']['ref'])
    spec=checked_file(ROOT,dcr['active_spec_ref'])
    require(digest(spec.read_bytes())==dcr['active_spec_sha256'], 'Active P01 spec hash mismatch')
    clock_dcr=document.get('dependency_clock_dcr')
    clock_approval=clock_dependency_state(document,clock_dcr) if clock_dcr else {}
    owners=('P01-A','P01-B','P01-B-HMB','P01-C','P01-D','P01-E','P01-F','P01-T')
    states=dict.fromkeys(owners,'NOT_STARTED')
    prerequisites={'P01-B-HMB':'P01-A','P01-C':'P01-B-HMB','P01-D':'P01-C',
                   'P01-E':'P01-D','P01-F':'P01-E'}
    technical_requirements={'P01-C':('C1','C2'),'P01-D':('D1',),
                            'P01-E':('E1',),'P01-F':('F1',)}
    accepted={}
    scopes={}
    previous=None
    for event in document['p01_subphase_events']:
        owner,status=event['phase'],event['to']
        moment=datetime.fromisoformat(event['at'])
        require(owner in states and moment.tzinfo is not None and (previous is None or moment>=previous),
                'Invalid P01 subphase chronology')
        require(event['from']==states[owner], 'P01 subphase replay mismatch')
        registration=owner=='P01-T' and states[owner]=='NOT_STARTED' and status=='BLOCKED'
        require(registration or status in EDGES[states[owner]], f'Invalid {owner} transition to {status}')
        prior=prerequisites.get(owner)
        if status=='IN_PROGRESS':
            require(bool(event.get('authorization_ref')), f'{owner} requires separate start authorization')
            if prior:
                require(states[prior]=='ACCEPTED', f'{owner} requires {prior} ACCEPTED')
            if owner in technical_requirements:
                require(moment>=approved and event.get('spec_sha256')==dcr['active_spec_sha256'],
                        f'{owner} requires the reapproved v0.2.1 spec')
            if owner=='P01-C':
                scopes[owner]=event.get('scope')
            if owner in ('P01-C','P01-D'):
                for field in ('implementation_spec','sample_plan'):
                    artifact=checked_file(ROOT,event[f'{field}_ref'])
                    require(digest(artifact.read_bytes())==event[f'{field}_sha256'],
                            f'{owner} frozen {field} differs')
                if owner=='P01-C' and event.get('scope')=='HMB_SIM_REVIEW':
                    require(clock_dcr and moment>=datetime.fromisoformat(clock_dcr['approved_at']) and
                            event['implementation_spec_sha256']==clock_dcr['spec_sha256'],
                            'P01-C simulation review requires the approved review spec')
        if status=='READY_FOR_REVIEW' and owner in ('P01-B-HMB','P01-T'):
            evidence=checked_file(ROOT,event['evidence_ref'])
            gate_key='HMB' if owner=='P01-B-HMB' else 'P01-T'
            require(digest(evidence.read_bytes())==event['evidence_sha256'] and
                    event.get('technical_gates',{}).get(gate_key)=='PASS',
                    f'{owner} requires a referenced technical PASS')
            payload=load_json(evidence)
            evidence_gate='new_hmb_gate' if owner=='P01-B-HMB' else 'technical_p01_t_gate'
            require(payload.get(evidence_gate)=='PASS', f'{owner} receipt does not support technical PASS')
        if (owner in technical_requirements and
                (status=='READY_FOR_REVIEW' or 'technical_gates' in event)):
            review_scope=event.get('scope')=='HMB_SIM_REVIEW'
            if owner=='P01-C':
                require(scopes.get(owner)==event.get('scope'),'P01-C exit scope differs from the authorized run')
            if review_scope:
                require(owner=='P01-C' and clock_dcr and moment>=datetime.fromisoformat(clock_dcr['approved_at']),
                        'Revised P01-C scope requires DCR-002 approval')
            evidence=checked_file(ROOT,event['evidence_ref'])
            require(digest(evidence.read_bytes())==event.get('evidence_sha256'),
                    f'{owner} technical receipt hash mismatch')
            receipt=load_json(evidence)
            require(receipt.get('phase')==owner and bool(receipt.get('run_id')),
                    f'{owner} technical receipt identity mismatch')
            gates=('C1-GEO','C2-SIM') if review_scope else technical_requirements[owner]
            require(not review_scope or receipt.get('scope')=='HMB_SIM_REVIEW',
                    'P01-C review receipt scope differs')
            if review_scope:
                separate=('C2-RT','robot-base','P01-T','Final_Test')
                require(receipt.get('verified_control_ticks')==0 and
                        all(receipt['technical_gates'].get(k)=='BLOCKED'==event['technical_gates'].get(k)
                            for k in separate),
                        'Bounded simulation review cannot grant independent hardware or final-test qualification')
            for gate in gates:
                actual=receipt.get('technical_gates',{}).get(gate)
                require(actual in {'PASS','FAIL','BLOCKED','NOT_RUN'} and
                        event.get('technical_gates',{}).get(gate)==actual,
                        f'{owner} receipt differs from logged {gate} outcome')
                require(status!='READY_FOR_REVIEW' or actual=='PASS',
                        f'{owner} receipt does not support {gate} technical PASS')
            if owner=='P01-D' and status=='READY_FOR_REVIEW':
                frozen=checked_file(ROOT,event['frozen_config_ref'])
                require(digest(frozen.read_bytes())==event['frozen_config_sha256']==receipt['candidate_config_sha256'],
                        'P01-D frozen config must match the reviewed exact bytes')
                require(load_json(frozen)['approval']=='PROPOSED_PENDING_HUMAN',
                        'P01-D technical gate does not grant human freeze approval')
        if status=='ACCEPTED':
            approval=event.get('approval',{})
            require(approval.get('type')=='human' and bool(approval.get('by')) and bool(approval.get('ref')),
                    f'{owner} requires human ACCEPTED')
            checked_file(ROOT,approval['ref'])
            if owner=='P01-C' and scopes.get(owner)=='HMB_SIM_REVIEW':
                require(event.get('scope')=='HMB_SIM_REVIEW','P01-C acceptance scope differs from review')
                path=event['evidence_ref']
                receipt=checked_file(ROOT,path)
                witness=event['evidence_sha256']
                require(digest(receipt.read_bytes())==witness,'P01-C final receipt hash mismatch')
                commit=event.get('reviewed_commit','')
                require(bool(re.fullmatch(r'[0-9a-f]{40}',commit)),'P01-C acceptance requires the reviewed commit')
                published=git('show',f'{commit}:docs/s0-5/{path}')
                require(published.returncode==0 and digest(published.stdout)==witness,
                        'Final receipt differs from reviewed P01-C commit')
                technical=load_json(receipt)
                require(technical['scope']=='HMB_SIM_REVIEW' and
                        all(technical['technical_gates'].get(k)=='PASS' for k in ('C1-GEO','C2-SIM')),
                        'P01-C approval requires successful bounded technical gates')
            accepted[owner]=moment
        else:
            accepted.pop(owner,None)
        states[owner],previous=status,moment
    require(states==document['p01_subphases'], 'P01 current subphase status differs from history')
    for event in document['events']:
        moment=datetime.fromisoformat(event['at'])
        if event['phase']=='P01' and event['to']=='ACCEPTED':
            require('P01-F' in accepted and accepted['P01-F']<=moment, 'P01 exit requires P01-F ACCEPTED')
        if event['phase']=='P03' and event['to']=='IN_PROGRESS':
            require('P01-T' in accepted and accepted['P01-T']<=moment, 'P03 requires P01-T ACCEPTED')
            if clock_dcr and moment>=datetime.fromisoformat(clock_dcr['approved_at']):
                require(all(q in clock_approval and clock_approval[q]<=moment for q in ('C2-RT','robot-base')),
                        'P03 requires C2-RT and robot-base ACCEPTED')
    print('PASS DCR-001 subphase replay, human gates, active spec and independent P03 dependency')


def clock_dependency_state(document,dcr):
    require(dcr['id']=='DCR-002' and dcr['approval']['type']=='human' and dcr['approval']['by'],
            'DCR-002 requires independent human dependency authorization')
    checked_file(ROOT,dcr['approval']['ref'])
    require(dcr['dependencies']=={'P01-C review':['C1-GEO','C2-SIM'],'P01-D':['P01-C'],
                                  'P03':['P02','P01-T','C2-RT','robot-base']},
            'DCR-002 dependencies differ')
    path=checked_file(ROOT,dcr['spec_ref'])
    require(digest(path.read_bytes())==dcr['spec_sha256'],'DCR-002 implementation spec hash mismatch')
    start=datetime.fromisoformat(dcr['approved_at'])
    require(start.tzinfo is not None,'DCR-002 requires a timezone-aware approval timestamp')
    states={'C2-RT':'BLOCKED','robot-base':'BLOCKED'}
    approved={}
    for event in document['clock_qualification_events']:
        key=event['qualification']
        when=datetime.fromisoformat(event['at'])
        require(key in states and states[key]=='BLOCKED' and event['status']=='ACCEPTED' and when>=start,
                'Unknown or out-of-order DCR-002 qualification')
        evidence=checked_file(ROOT,event['evidence_ref'])
        require(digest(evidence.read_bytes())==event['evidence_sha256'] and
                load_json(evidence).get('technical_gates',{}).get(key)=='PASS',
                f'{key} needs independently hashed technical PASS')
        require(event['approval']['type']=='human' and event['approval']['by'],f'{key} needs human approval')
        checked_file(ROOT,event['approval']['ref'])
        approved[key]=when
        states[key]='ACCEPTED'
    require(document['clock_qualifications']==states,'DCR-002 qualification state differs from history')
    return approved


def rows(path):
    return [[v.strip() for v in line.strip('|').split('|')] for line in read(path).splitlines()
            if line.startswith('| S05-REQ-')]


def anchors(path):
    titles = re.findall(r'^#{1,6} (.+)$', read(path), flags=re.M)
    return {re.sub(r'[^\w -]', '', title.lower()).replace(' ', '-') for title in titles}


def checked_links(files):
    total = 0
    for src in files:
        for url in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', read(src)):
            if url.startswith(('https://','http://')):
                continue
            name, _, anchor = url.partition('#')
            target = (src.parent / name).resolve() if name else src
            require(target.is_relative_to(ROOT) and target.exists(), f'{src.name}: {url}')
            require(not anchor or target.suffix != '.md' or anchor in anchors(target), f'Broken heading: {url}')
            total += 1
    return total


def structure():
    files = list(ROOT.rglob('*.md'))
    for name in TOP:
        checked_file(ROOT, f'{name}.md')
    checked_file(ROOT, 'reports/P00_acceptance.md')
    checked_file(ROOT, 'reports/P00_review_fix.md')
    req = rows(ROOT / 'REQUIREMENTS.md')
    matrix = rows(ROOT / 'TRACEABILITY_MATRIX.md')
    ids = [r[0] for r in req]
    require(len(ids) == len(set(ids)) and [r[0] for r in matrix] == ids, 'Duplicate or missing requirement ID')
    require({f'S05-REQ-{i:03}' for i in range(1,31)} <= set(ids), 'Core requirement IDs missing')
    require(all(len(r)==4 and all(r) and r[2] in PHASES for r in req), 'Requirement row schema')
    require(all(len(r)==9 and all(r) for r in matrix) and [r[2] for r in req]==[r[3] for r in matrix],
            'Traceability row/ownership')
    p01_adjuncts={'P01_T_telemetry_acquisition.md','P01_C_calibration_and_causal_sync.md',
                  'P01_C_RT_clock_qualification.md','P01_C_simulation_review_v0.2.0.md',
                  'P01_D_dataset_freeze.md'}
    specs = sorted(s for s in (ROOT / 'specs').glob('P??_*.md') if s.name not in p01_adjuncts)
    require(tuple(s.name[:3] for s in specs) == PHASES, 'Phase spec list')
    for spec in specs:
        doc = read(spec)
        sections = [int(x) for x in re.findall(r'^## (\d+)\. ', doc, re.M)]
        require(sections == list(range(1,26)), f'Incomplete phase spec: {spec.name}')
        cover = doc.split('## 4. ',1)[1].split('## 5. ',1)[0]
        for req_id, _, owner, _ in req:
            if owner == spec.name[:3]:
                require(req_id[-3:] in cover, f'{spec.name} does not cite {req_id}')
    print(f'PASS structure: {len(files)} Markdown files, {len(ids)} requirements, '
          f'{len(specs)} phase specs, {checked_links(files)} local links')


def json_pointer(document, pointer):
    require(pointer.startswith('/'), f'Invalid JSON pointer {pointer}')
    value = document
    for item in pointer[1:].split('/'):
        key = item.replace('~1','/').replace('~0','~')
        value = value[int(key)] if isinstance(value,list) else value[key]
    return value


def checked_manifest(root, manifest):
    require(manifest['schema'] == 's05.review-evidence.v1', 'Review evidence manifest schema')
    evidence = {}
    for entry in manifest['files']:
        rel = entry['path']
        path = checked_file(root, rel)
        content = path.read_bytes()
        require(rel not in evidence and digest(content) == entry['sha256'] and len(content) == entry['size'],
                f'Invalid or changed evidence: {rel}')
        evidence[rel] = path
    return evidence


def recorded_findings():
    env, samples, historical = (load_json(REVIEW / f'{name}.json') for name in
                                ('environment','samples','historical'))
    for name, doc in zip(('environment','samples','historical'), (env,samples,historical),strict=True):
        require(doc['schema']=='s05.readiness.witness.v1' and doc['section']==name and
                datetime.fromisoformat(doc['observed']).tzinfo is not None, f'Bad witness identity: {name}')
    runtime = env['result']
    require(not runtime['wrong_dexgym'] and not runtime['foreign'], 'Historical code leaked into parser imports')
    output = runtime['output']
    require(output['writable'] and output['distinct_from_source'] and output['available_bytes']>0,
            'Missing independent output witness')
    require(all(v['status']=='PASS' and HEX.fullmatch(v['file']['sha256']) for v in runtime['imports'].values())
            and runtime['overlays']['ROS_DISTRO']=='jazzy', 'Incomplete parser import evidence')
    groups = historical['result']['groups']
    require(set(groups)=={'wholebody_s0','wholebody_s0_1','wholebody_s0_2','marvin54','wholebody_s04'},
            'Incomplete historical registry probe')
    data = samples['result']
    require(data['native_manifest']['top_level_scalars']['format']=='dex-native-v2', 'Unexpected native format')
    require(all(data['parsed_schema'][f'manus_{side}__pose']['shape'][1]==25 for side in ('left','right')),
            'Historic MANUS fixture must retain its observed 25 nodes')
    require(data['merged']['publish_time'].startswith('UNKNOWN') and data['merged']['controller_receive_time']=='UNKNOWN',
            'Index reader timestamp witness has changed semantics')
    require(all(v['count']>0 and v['count']<=v['limit'] for v in data['wire_clocks'].values()), 'Bounded MCAP clock witness')
    require(all(data['merged']['decoded'][f'/manus_glove_{side}']['count']>0 for side in ('left','right')),
            'Missing legacy MANUS CDR witness')
    legacy = data['merged_embedded_schema']['topics']
    require(all(legacy[f'/{side}_hand/state']['count']>0 and
                legacy[f'/{side}_hand/state']['schema_encoding']=='ros2msg' for side in ('left','right')),
            'Missing embedded-schema fallback witness for legacy hands')


def evidence_and_refs():
    evidence = checked_manifest(ROOT,load_json(REVIEW / 'review_evidence_manifest.json'))
    runs = load_json(REVIEW / 'commands.json')
    require(runs['schema'] == 's05.review-commands.v1' and bool(runs['commands']), 'Command evidence schema')
    for command in runs['commands']:
        require(bool(command['command']) and bool(command['cwd']), 'Command identity missing')
        outcome = command['status']
        require(outcome in {'PASS','FAILED','BLOCKED','NOT_RUN'}, 'Unknown command result')
        if outcome == 'PASS':
            require(command['exit'] == 0, 'A passing command must have exit 0')
        elif outcome == 'FAILED':
            require(isinstance(command['exit'],int) and command['exit'] != 0, 'A failed command needs its exit')
        else:
            require(bool(command.get('reason')), 'Blocked/unrun command needs a reason')
        for ref in command.get('artifacts',[]):
            require(ref in evidence, f'Unhashed command artifact {ref}')
        if ref := command.get('exit_artifact'):
            require(ref in evidence and int(read(evidence[ref]).strip()) == command['exit'],
                    f'Exit artifact mismatch for {command["id"]}')
    index = load_json(REVIEW / 'review_gates.json')
    require(index['schema'] == 's05.review-gates.v1', 'Review gate index schema')
    gates = index['gates']
    require([g['id'] for g in gates] == [f'R{i:02}' for i in range(1,7)], 'Missing review gate')
    for gate in gates:
        require(gate['status'] in {'PASS','FAIL','BLOCKED','NOT_RUN'} and gate['references']
                and bool(gate['remaining_risk']), f'Incomplete row {gate["id"]}')
        for ref in gate['references']:
            path = checked_file(ROOT,ref['path'])
            if ref['kind']=='recorded_evidence':
                require(ref['path'] in evidence, f'Unhashed gate evidence {path}')
                if ref.get('pointer'):
                    require(json_pointer(load_json(path),ref['pointer']) is not None, f'Unknown evidence field in {path}')
            else:
                require(ref['kind']=='design' and path.suffix=='.md', f'Invalid evidence role {path}')
    report = read(ROOT / 'reports/P00_review_fix.md')
    lines = {m.group(1):m.group(2).strip() for m in re.finditer(
        r'^\| (R\d{2}) \|[^|\n]+\|[^|\n]+\| ([^|\n]+)\|',report,re.M)}
    require(lines == {g['id']:g['status'] for g in gates}, 'Report and gate index differ')
    recorded_findings()
    phase_states(load_json(ROOT/'phase_state.json'))
    print(f'PASS integrity: {len(evidence)} hashed evidence files and {len(gates)} referenced review rows')
    print('PASS phase event log; approval authenticity and technical acceptance require human review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-history', action='store_true')
    parser.add_argument('--history-only', action='store_true')
    args = parser.parse_args()
    historical_snapshot(args.require_history or args.history_only)
    if not args.history_only:
        structure()
        evidence_and_refs()


if __name__ == '__main__':
    main()
