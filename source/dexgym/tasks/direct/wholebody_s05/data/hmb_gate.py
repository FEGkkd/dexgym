"""Evaluate the v0.2.1 HMB decode scope without changing training eligibility."""
from collections import Counter,defaultdict
import math
import re


HEX = re.compile(r'[0-9a-f]{64}')
CLASSES = {'HMB_REQUIRED','HMB_CONDITIONAL','TELEMETRY_REQUIRED','AUXILIARY_DIAGNOSTIC','OUT_OF_SCOPE'}


def dependency_index(matrix):
    index = {}
    for rule in matrix['streams']:
        for topic in rule['topics']:
            if topic in index or rule['class'] not in CLASSES:
                raise ValueError(f'Ambiguous dependency rule for {topic}')
            index[topic] = rule
    return index


def pinned_index(index, parts, topic, schema):
    return (index.get('topic')==topic and index.get('schema_sha256')==schema and
            index.get('asset_sha256')==parts.get(index.get('part_id')) and
            bool(HEX.fullmatch(index.get('asset_sha256') or '')) and
            isinstance(index.get('physical_message'),int) and index['physical_message']>=0)


def clock(raw, name):
    return raw.get('clock',{}).get(name,{}).get('value')


def finite_vector(vector,size):
    return (isinstance(vector,list) and len(vector)==size and
            all(type(v) in {int,float} and math.isfinite(v) for v in vector))


def validates_shape(event,role):
    fields = event.get('fields') or {}
    if role=='manus':
        h = event.get('manus',{})
        nodes = h.get('nodes') or []
        ids = {node.get('node_id') for node in nodes}
        return (bool(nodes) and h.get('node_count')==fields.get('raw_node_count')==len(nodes) and
                not h.get('issues') and len(ids)==len(nodes) and
                h.get('original_node_order')==list(range(len(nodes))) and
                all(n.get('original_order')==i and type(n['node_id']) is int and
                    type(n.get('parent_node_id')) is int and
                    (n['parent_node_id'] in ids or n['parent_node_id']==-1) and
                    isinstance(n.get('joint_type'),str) and isinstance(n.get('chain_type'),str) and
                    finite_vector(n.get('node_position'),3) and finite_vector(n.get('node_orientation'),4)
                    for i,n in enumerate(nodes)) and
                all(type(clock(event,c)) is int and clock(event,c)>0 for c in ('sdk_source','sdk_receive')))
    if role=='tracker_safety':
        return type(fields.get('safe')) is bool and isinstance(fields.get('code'),str) and bool(fields['code'])
    pose = fields.get('pose') or {}
    position,rotation = pose.get('position') or {}, pose.get('orientation') or {}
    expected = event['index']['topic'].rsplit('/',1)[-1]
    valid = role!='registered_tracker' or (
        type(fields.get('valid')) is bool and fields.get('role')==expected and
        type(fields.get('quality')) in {int,float} and math.isfinite(fields['quality']))
    return (valid and bool((fields.get('header') or {}).get('frame_id')) and
            finite_vector([position.get(k) for k in 'xyz'],3) and
            finite_vector([rotation.get(k) for k in 'xyzw'],4))


def observation_pair(event, other, rule, parts, collector):
    """Check both indexes of the writer-attested pair, including raw clock values."""
    pair = event.get('pairing') or {}
    ptr, original = pair.get('observation_index') or {},event['index']
    obs_spec = rule['observation']
    stream = rule['topics'].index(original['topic'])
    topic = obs_spec['topics'][stream]
    if (not other or other.get('segment')!='BODY' or other.get('decode_status')!='PASS' or
            other.get('index')!=ptr or not pinned_index(ptr,parts,topic,obs_spec['schema_sha256']) or
            original['part_id']!=ptr['part_id'] or ptr['physical_message']!=original['physical_message']+1 or
            pair.get('status')!='VERIFIED_WRITER_PAIR' or pair.get('collector_instance_id')!=collector):
        return False
    obs = other['fields']
    return (bool(pair.get('source_instance_id')) and obs.get('source_instance_id')==pair['source_instance_id'] and
            obs.get('stream_id')==topic.rsplit('/',1)[-1] and
            type(obs.get('episode_sample_index')) is int and
            obs['episode_sample_index']==pair.get('episode_sample_index') and
            type(obs.get('source_sequence_id')) is int and obs['source_sequence_id']==pair.get('source_sequence_id') and
            type(clock(event,'mcap_log')) is int and type(clock(event,'mcap_publish')) is int and
            type(clock(event,'ros_header')) is int and
            clock(event,'mcap_log')==clock(other,'mcap_log')==obs.get('host_realtime_ns') and
            all(type(obs.get(field)) is int and clock(event,key)==clock(other,key)==obs[field]
                for key,field in (('collector_monotonic','host_observed_monotonic_ns'),
                                  ('collector_receive','receive_ros_timestamp_ns'),
                                  ('collector_realtime','host_realtime_ns'))))


def evaluated_fixture(fixture, events, registry, source, matrix, calibrations_ok):
    """Return a bounded capability decision. Excluded sources never become candidates."""
    ident = fixture['recording_id']
    admitted = (registry.get('status')=='ADMITTED_CANDIDATE' and
                registry.get('producer_kind')=='LIVE_TELEOP' and
                source.get('raw_human_source_verified') is True and
                source.get('level')=='RAW_SOURCE_CONFIRMED_FIELDS_SEPARATE' and
                registry.get('id')==source.get('recording_id')==ident and
                bool(registry.get('root_episode_id')) and
                registry['root_episode_id']==fixture.get('root_episode_id'))
    outcome = dict(recording_id=ident,format=fixture['format'],formal_training_eligible=False,
                   role='BOUNDED_DIAGNOSTIC',status='ISOLATED_SOURCE',reasons=[],counts={})
    if not admitted:
        outcome['reasons'].append('No joint A1 and source eligibility for raw HMB')
        return outcome
    if (fixture['format']!=matrix['admission_format'] or not fixture.get('verified_writer') or
            not calibrations_ok or not registry.get('collector_instance')):
        outcome.update(status='BLOCKED',reasons=['Missing admitted writer, collector or calibration provenance'])
        return outcome

    body = [e for e in events if e.get('segment')=='BODY']
    by_topic = defaultdict(list)
    locations = {}
    for e in body:
        i = e['index']
        key = (i['part_id'],i['physical_message'])
        if key in locations:
            outcome.update(status='FAIL',reasons=['Duplicate physical record index'])
            return outcome
        locations[key] = e
        by_topic[i['topic']].append(e)
    rules = dependency_index(matrix)
    parts = {p['part_id']:p['sha256'] for p in fixture['parts']}
    failures, missing = [], []
    safety_unknown = 0
    for topic,rule in rules.items():
        if rule['class']!='HMB_REQUIRED':
            continue
        rows = by_topic[topic]
        outcome['counts'][topic] = len(rows)
        if not rows:
            missing.append(f'{topic}: no BODY messages')
        segments = defaultdict(list)
        for row in rows:
            i = row['index']
            m = f"{topic}@{i['part_id']}:{i['physical_message']}"
            if (row.get('decode_status')!='PASS' or row.get('nonfinite') or
                    not pinned_index(i,parts,topic,rule['schema_sha256']) or
                    not HEX.fullmatch(row.get('cdr_sha256') or '') or
                    not validates_shape(row,rule['role'])):
                failures.append(f'{m}: required raw decode, schema or fields')
                continue
            obs = (row.get('pairing') or {}).get('observation_index') or {}
            paired = locations.get((obs.get('part_id'),obs.get('physical_message')))
            if not observation_pair(row,paired,rule,parts,registry['collector_instance']):
                missing.append(f'{m}: physical writer pair or raw timestamp unavailable')
                continue
            pair = row['pairing']
            segments[pair['source_instance_id']].append(
                (pair['episode_sample_index'],pair['source_sequence_id'],clock(row,'collector_monotonic')))
            if rule['role']=='tracker_safety':
                safety_unknown += int(row['fields']['safe'] is False)
        for samples in segments.values():
            # Bounded prefixes preserve index and sequence, with known drops allowed.
            if any(b[0]!=a[0]+1 or b[1]<=a[1] or b[2]<a[2]
                   for a,b in zip(samples,samples[1:])):
                missing.append(f'{topic}: inconsistent writer episode index, sequence or monotonic clock')

    # An explicit dependency on an auxiliary channel is stronger than its generic class.
    for topic in fixture.get('required_proof_topics',[]):
        if (topic not in rules or not by_topic[topic] or any(
                e.get('decode_status')!='PASS' or not e.get('fields') for e in by_topic[topic])):
            missing.append(f'{topic}: declared evidence is unavailable')
    outcome.update(status='FAIL' if failures else 'BLOCKED' if missing else 'PASS',
                   reasons=failures+missing,known_unsafe_safety_samples=safety_unknown)
    return outcome


def evaluate(fixtures, events, registry, source, matrix, calibration_ids):
    by_fixture = defaultdict(list)
    for event in events:
        by_fixture[event['recording_id']].append(event)
    outcomes = [
        evaluated_fixture(f,by_fixture[f['recording_id']],registry[f['recording_id']],
                          source[f['recording_id']],matrix,f['recording_id'] in calibration_ids)
        for f in fixtures
    ]
    counts = Counter(r['status'] for r in outcomes)
    gate = ('FAIL' if counts['FAIL'] else 'BLOCKED' if counts['BLOCKED'] or not counts['PASS'] else 'PASS')
    return {'technical_hmb_gate':gate,'fixture_counts':dict(counts),'fixtures':outcomes,
            'admission_effect':'NONE; this gate checks bounded raw decode witnesses',
            'physical_validation':'NOT_RUN_P01_C'}
