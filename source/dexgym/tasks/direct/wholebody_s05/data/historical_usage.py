"""Record historic split declarations and evidence of actual use separately."""
from collections import defaultdict

SPLIT_ROLES = {
    'train': 'TRAIN', 'training': 'TRAIN', 'val': 'VAL', 'validation': 'VAL',
    'test': 'HISTORICAL_TEST_ASSIGNMENT', 'heldout': 'HISTORICAL_TEST_ASSIGNMENT',
}
ROLE_KEYS = {
    'model_selection': 'MODEL_SELECTION', 'checkpoint_selection': 'MODEL_SELECTION',
    'hard_cases': 'HARD_CASE_ANALYSIS', 'hard_sequences': 'HARD_CASE_ANALYSIS',
    'repeated_test': 'REPEATED_TEST', 'tuning': 'TUNING', 'calibration': 'CALIBRATION',
    'integrity_only': 'INTEGRITY_ONLY', 'calibration_episodes': 'CALIBRATION',
}
SOURCE_KEYS = ('source_episode', 'source_episode_id', 'episode_id', 'raw_episode_id',
               'parent_episode_id', 'episode', 'source_recording_id', 'recording_id',
               'origin_episode', 'source_clip')
SPAN_KEYS = ('source_start_frame', 'source_stop_frame_exclusive', 'start_frame', 'end_frame',
             'start_time_ns', 'duration_s', 'frame_start', 'frame_end')
PARENT_KEYS = ('parent_ids', 'source_episode_ids', 'source_episodes', 'parent_episodes',
               'parent_recording_ids', 'source_recordings', 'parent_clips')
ARTIFACT_KEYS = ('source_manifest', 'parent_manifest', 'dataset_manifest', 'reference_manifest',
                 'library_manifest', 'manifest_path', 'dataset_path', 'bank_manifest', 'source_library')


def history_events(document, artifact, family):
    """Parse named metadata fields, preserving source interval and evidence path."""
    result = []
    for node, field, role in iter_context(document):
        refs = references(node)
        if not refs or (role is None and not any(k in node for k in SOURCE_KEYS + PARENT_KEYS)):
            continue
        span = {k:node[k] for k in SPAN_KEYS if isinstance(node.get(k), (int, float))}
        for key in ('source_frame_interval','collision_parent_frame_interval','timing'):
            value = node.get(key)
            if isinstance(value,(dict,list)):
                span[key] = value
        for ref in refs:
            result.append(dict(schema='historical_usage.v1', source_ref=ref, family=family,
                               artifact_id=artifact, evidence_field=field or '/',
                               role=role or 'USAGE_UNKNOWN',
                               confidence=('DECLARED_SPLIT' if role in SPLIT_ROLES.values() else
                                           'DECLARED_CONTEXT' if role else 'UNRESOLVED'), interval=span))
    return result


def references(node):
    refs = {node[k] for k in SOURCE_KEYS if isinstance(node.get(k),str) and node[k]}
    for key in PARENT_KEYS:
        value = node.get(key)
        if isinstance(value,list):
            refs.update(x for x in value if isinstance(x,str) and x)
    return sorted(refs)


def iter_context(value, field='', role=None):
    if isinstance(value,list):
        for i,item in enumerate(value):
            yield from iter_context(item,f'{field}/{i}',role)
    elif isinstance(value,dict):
        split = value.get('split')
        role = SPLIT_ROLES.get(split.lower(),role) if isinstance(split,str) else role
        yield value, field, role
        for name, item in value.items():
            if not isinstance(item,(list,dict)):
                continue
            extra = ROLE_KEYS.get(name)
            split_role = SPLIT_ROLES.get(name) if field.endswith('/splits') else None
            yield from iter_context(item,f'{field}/{name}',extra or split_role or role)


def root_use(events, graph):
    result = defaultdict(set)
    for event in events:
        roots,_ = graph.resolve(event['source_ref'])
        for root in roots:
            result[root].add(event['role'])
    return {root:sorted(roles) for root,roles in sorted(result.items())}
