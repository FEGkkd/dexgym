"""Structured P01-E qualification comparing raw evidence to original D1 approval."""
from dataclasses import dataclass
import json
from math import isfinite

from .synchronize import STREAMS


@dataclass(frozen=True)
class Qualification:
    status: str
    hard: tuple
    insufficient: tuple
    retained: tuple
    pairs: dict
    inherited_from: str | None = None

    @property
    def eligible(self):
        return self.status in {'ELIGIBLE', 'TAIL_INHERITED'}


def _residual_issues(pair, limits):
    result=[]
    side=pair['side']
    for failure in pair['failures']:
        if failure['reason']!='REGISTERED_INVALID':
            result.append(f'{side}:FRAME_OR_CALIBRATION_FAILURE')
    for column,unit in (('translation_m','m'),('rotation_rad','rad')):
        peak=pair[column]['max']
        if pair['matched'] and (not isinstance(peak,(int,float)) or not isfinite(peak) or peak<0 or peak>limits[unit]):
            result.append(f'{side}:CALIBRATION_RESIDUAL_EXCEEDED:{unit}')
    return result


def qualify_partition(source, witness, block, limits, allowed_generations, inherited=None):
    """Derive decision from typed fields. Human-readable D1 issues are diagnostic only."""
    hard=[]
    insufficient=[]
    retained=[]
    p=witness['production_witness']
    binding=('recording_id','root_episode_id','episode_id','collector_instance_id','manifest_sha256',
             'snapshot_sha256','calibration_sha256','assets')
    if any(source[key]!=witness[key] for key in binding):
        hard.append('SOURCE_SNAPSHOT_CALIBRATION_BINDING_MISMATCH')
    if set(witness['source_read'])!=set((*STREAMS,'tracker_safety')):
        insufficient.append('REQUIRED_STREAM_EVIDENCE_ABSENT')
    if any(x['invalid_schema_rows'] for x in witness['source_read'].values()):
        hard.append('SCHEMA_OR_REQUIRED_GEOMETRY_FIELD_INVALID')
    if p['invalid_geometry']:
        hard.append('GEOMETRY_UNIT_OR_COORDINATE_UNPROVEN')
    fingerprints={json.dumps(t,sort_keys=True) for t in source['topologies']}
    if any(json.dumps(t,sort_keys=True) not in fingerprints for t in witness['topologies']):
        hard.append('TOPOLOGY_NOT_APPROVED')
    if {t['stream'] for t in witness['topologies']}!={'manus_left','manus_right'}:
        insufficient.append('MANUS_TOPOLOGY_EVIDENCE_ABSENT')
    if (witness['capture_qualification']!='SIMULATED' or
            any(w['max_abs_recorder_residual_ns']>limits['archived_clock_pair_limit_ns'] for w in witness['run_clocks']) or
            not p['observed_clock_generations'] or not set(p['observed_clock_generations'])<=set(allowed_generations) or
            any(p['monotonic_regressions'].values())):
        hard.append('TIME_OR_CLOCK_GENERATION_UNAPPROVED')
    if p['known_registered_invalid']:
        retained.append('REGISTERED_INVALID_SAMPLES')
    local=p['body_calibration'].get(block['label'],[])
    for pair in witness['calibrated_side_pairs']+local:
        hard.extend(_residual_issues(pair,limits['calibration_error_limits']))
        if any(pair[key] for key in ('raw_unpaired','registered_unpaired','ambiguous_header_stamps')):
            retained.append(f"{pair['side']}:UNPAIRED_OR_DUPLICATE_OBSERVATION")
    pairs={a['side']:a['matched'] for a in local}
    if set(pairs)!={'left','right'} or min(pairs.values(),default=0)<limits['min_registered_pairs_per_side']:
        insufficient.append('LOCAL_SOFTWARE_CALIBRATION_PAIR_COUNT')
    hard=tuple(sorted(set(hard)))
    insufficient=tuple(sorted(set(insufficient)))
    retained=tuple(sorted(set(retained)))
    if hard:
        return Qualification('HARD_FAILURE',hard,insufficient,retained,pairs)
    if not insufficient:
        return Qualification('ELIGIBLE',hard,insufficient,retained,pairs)
    if (insufficient==('LOCAL_SOFTWARE_CALIBRATION_PAIR_COUNT',) and block['tail'] and
            block['stop_ns']-block['start_ns']<=block['tail_inheritance_max_ns'] and inherited and
            inherited['recording_id']==source['recording_id'] and inherited['scope_digest']==block['scope_digest'] and
            inherited['classification']=='ELIGIBLE' and inherited['full_block'] and
            inherited['stop_ns']<=block['start_ns']):
        return Qualification('TAIL_INHERITED',hard,insufficient,retained,pairs,inherited['manifest_sha256'])
    return Qualification('INSUFFICIENT',hard,insufficient,retained,pairs)
