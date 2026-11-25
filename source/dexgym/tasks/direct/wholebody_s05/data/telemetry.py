"""Separate telemetry field observations from proven hardware and pairing eligibility."""
from collections import Counter
import json
from pathlib import Path

from .registry_gate import digest
from .raw_events import header_value, stamp

ARM = ('left','right')
TOPICS = {
    '/tianji/joint_command':('arm_command',14),
    '/tianji/robot_state':('driver_state',None),
    '/tianji/teleop_state':('teleop_state',None),
    '/tianji/teleop_diagnostics':('teleop_diagnostics',None),
    '/tianji/safety':('arm_safety',None),
    '/tianji_real_driver/safety':('real_driver_safety',None),
    '/vive/tracker/safety_status':('tracker_safety',None),
}
for side in ARM:
    TOPICS.update({
        f'/{side}_arm/joint_states':(f'{side}_arm_feedback',7),
        f'/{side}_arm/ee_pose':(f'{side}_wrist_pose',None),
        f'/{side}_arm/ee_twist':(f'{side}_wrist_twist',None),
        f'/{side}_arm/wrench':(f'{side}_wrist_wrench',None),
        f'/{side}_hand/command':(f'{side}_hand_command',20),
        f'/{side}_hand/state':(f'{side}_hand_feedback',20),
        f'/{side}_hand/diagnostics':(f'{side}_hand_diagnostics',None),
        f'/{side}_hand/safety':(f'{side}_hand_safety',None),
    })


def verify_attestation(recording, evidence, source_roots, ordered_asset_sha256):
    """Require prior review and capture-bound independent hashes; topics are never attestations."""
    if (not evidence or evidence.get('recording_id')!=recording['id'] or not evidence.get('review_ref') or
            not ordered_asset_sha256 or any(not sha for sha in ordered_asset_sha256) or
            evidence.get('ordered_asset_sha256')!=ordered_asset_sha256):
        return {}
    kind = evidence.get('kind')
    if kind not in {'LIVE_TELEOP','REAL_ROBOT_TELEOP','SIM_REPLAY','DERIVED_ONLY','EGOPRO'}:
        return {}
    required = {'capture_config','recorder_identity','original_binding'}
    if kind=='REAL_ROBOT_TELEOP':
        required |= {'hardware_identity','driver_log'}
    verified = {}
    for entry in evidence.get('artifacts',[]):
        path = Path(entry['path']).resolve()
        if any(path.is_relative_to(Path(root).resolve()) for root in source_roots) and path.is_file():
            if digest(path)==entry.get('sha256'):
                verified[entry['role']]={'path':str(path),'sha256':entry['sha256']}
    if not required<=verified.keys() or len({v['sha256'] for v in verified.values()})<len(required):
        return {}
    return {'kind':kind,'verified_artifacts':verified,'review_ref':evidence['review_ref']}


def source_eligibility(recording, attested):
    evidence = attested if isinstance(attested,dict) else {}
    origin = recording.get('producer_kind','UNKNOWN')
    origin = origin if origin not in {'LIVE_TELEOP','UNKNOWN'} else evidence.get('kind',origin)
    status = recording.get('status','QUARANTINED')
    live = origin in {'LIVE_TELEOP','REAL_ROBOT_TELEOP'} and (
        status=='ADMITTED_CANDIDATE' or bool(evidence))
    if status=='DUPLICATE_ALIAS':
        level = 'DUPLICATE_ALIAS'
        live = False
    elif origin in {'SIM_REPLAY','EGOPRO','DERIVED_ONLY'}:
        level = 'EXCLUDED'
    elif live:
        level = 'RAW_SOURCE_CONFIRMED_FIELDS_SEPARATE'
    else:
        level = 'QUARANTINED'
    return dict(origin=origin,level=level,raw_human_source_verified=live,
                hardware_provenance=bool(live and evidence.get('kind')=='REAL_ROBOT_TELEOP'),
                formal_training_eligible=False,attestation=evidence or None)


def array_info(fields, keys):
    for key in keys:
        if key in fields:
            value = fields[key]
            return {'field':key,'length':len(value) if isinstance(value,list) else None,'value':value}
    return {'field':None,'length':None,'value':None}


def telemetry_row(event):
    topic = event['index']['topic']
    if topic not in TOPICS or event['decode_status']!='PASS':
        return None
    role,dim = TOPICS[topic]
    fields = event['fields']
    names = array_info(fields,('joint_names','name'))
    position = array_info(fields,('position','positions'))
    vectors = {
        'position':position,'velocity':array_info(fields,('velocity','velocities')),
        'effort':array_info(fields,('effort','efforts')),'current':array_info(fields,('current','currents')),
    }
    name_list = names['value']
    unique_ordered_names = (isinstance(name_list,list) and all(isinstance(x,str) and x for x in name_list)
                            and len(name_list)==len(set(name_list)))
    side = role.split('_',1)[0]
    side_match = 'side' not in fields or side not in ARM or str(fields['side']).lower()==side
    finite_position = isinstance(position['value'],list) and all(
        isinstance(x,(int,float)) and not isinstance(x,bool) for x in position['value'])
    has_axes = (dim is not None and unique_ordered_names and names['length']==dim and
                position['length']==dim and finite_position and side_match)
    validity = {key:fields[key] for key in ('connected','enabled','state','position_valid','velocity_valid',
                                           'effort_valid','online','stale','feedback_source') if key in fields}
    if role.endswith('_feedback') and fields.get('source_timestamp_valid') is True:
        feedback = stamp(fields.get('source_timestamp_us'),fields.get('source_clock') or None,
                         'us' if fields.get('source_clock') else None,'source_timestamp_us')
    else:
        # A ROS header on a state message can be publication time.
        feedback = stamp(None)
    clock = dict(event['clock'],robot_feedback=feedback)
    return dict(recording_id=event['recording_id'],index=event['index'],role=role,
                expected_dimension=dim,names=names,field_vectors=vectors,axes_match_recorded_schema=bool(has_axes),
                side_consistent=side_match,available_fields=sorted(fields),validity=validity,clock=clock,
                effort_unit=fields.get('effort_unit'),position_unit=('rad for verified revolute joints'
                if 'sensor_msgs/msg/JointState|' in (event['index'].get('schema_key') or '') else None),
                driver_sequence=fields.get('sequence',fields.get('source_sequence')),
                measured_or_command='COMMAND' if role.endswith('_command') else
                                    'FEEDBACK_TOPIC_UNVALIDATED_ORIGIN' if role.endswith('_feedback') else 'AUXILIARY')


PAIRS = {
    'left_arm_feedback':'arm_command', 'right_arm_feedback':'arm_command',
    'left_hand_feedback':'left_hand_command', 'right_hand_feedback':'right_hand_command',
}


def validate_bindings(recording_id, rows, attested):
    """Tie evidence-reviewed driver pairs to the actual raw records and named axes."""
    assets = (attested or {}).get('verified_artifacts',{})
    witness = assets.get('pairing_witness')
    if not witness:
        return False,False
    rows_by_key = {(r['index']['part_id'],r['index']['physical_message']):r for r in rows}
    pairing = json.loads(Path(witness['path']).read_bytes())
    if pairing.get('recording_id')!=recording_id:
        return False,False
    validated = set()
    keys = set()
    for item in pairing.get('verified_pairs',[]):
        cmd_key, feedback_key = (tuple(item.get(role,[])) for role in ('command','feedback'))
        cmd, response = rows_by_key.get(cmd_key),rows_by_key.get(feedback_key)
        if not cmd or not response or PAIRS.get(response['role'])!=cmd['role']:
            continue
        if not cmd['axes_match_recorded_schema'] or not response['axes_match_recorded_schema']:
            continue
        command_names, measured_names = cmd['names']['value'],response['names']['value']
        if set(measured_names)<=set(command_names):
            validated.add(response['role'])
            keys.add((cmd_key,feedback_key))
    full_pairing = set(PAIRS)<=validated
    clock_provenance = assets.get('clock_witness')
    if not full_pairing or not clock_provenance:
        return full_pairing,False
    clock = json.loads(Path(clock_provenance['path']).read_bytes())
    # Clock evidence must be capture-bound and assert a reviewed error bound; no timeline is fitted here.
    time_keys = {(tuple(r.get('command',[])),tuple(r.get('feedback',[])))
                 for r in clock.get('verified_pairs',[])
                 if isinstance(r.get('clock_error_upper_ns'),int) and r['clock_error_upper_ns']>=0
                 and r.get('controller_receive_source') and r.get('physical_feedback_source')}
    return full_pairing,bool(clock.get('recording_id')==recording_id and keys<=time_keys)


def eligibility(hardware_verified, roles, pair_verified, clocks_verified):
    required = set(PAIRS)|set(PAIRS.values())
    if not hardware_verified:
        level = 'NO_HARDWARE_PROVENANCE'
    elif not pair_verified and all(roles.get(k) for k in PAIRS):
        level = 'STATIC_AUDIT_CANDIDATE'
    elif not required<=roles.keys() or not pair_verified or not all(roles[k] for k in required):
        level = 'UNVERIFIED_PAIR'
    elif not clocks_verified:
        level = 'RELATIVE_RESPONSE_CANDIDATE'
    else:
        level = 'DYNAMIC_SYSTEM_ID_CANDIDATE'
    return {'level':level,'dynamic_system_id_candidate':level=='DYNAMIC_SYSTEM_ID_CANDIDATE',
            'hardware_provenance':hardware_verified,'command_feedback_pair':pair_verified,
            'time_evidence':clocks_verified,'observed_roles':sorted(roles)}


def observed_roles(rows):
    flags = {}
    for row in rows:
        if row['expected_dimension'] is not None:
            flags[row['role']] = flags.get(row['role'],True) and row['axes_match_recorded_schema']
    return flags
