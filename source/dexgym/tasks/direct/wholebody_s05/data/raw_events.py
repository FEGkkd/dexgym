"""Raw event evidence, typed clocks, and MANUS geometry without remapping."""
from hashlib import sha256
import math

from .source_registry import stable_json

CLOCKS = ('source_capture','sdk_source','sdk_receive','ros_header','mcap_log','mcap_publish',
          'collector_receive','collector_monotonic','collector_realtime','controller_receive',
          'command_issue','robot_feedback')


def stamp(value, domain=None, unit=None, field=None):
    seen = isinstance(value,int) and not isinstance(value,bool)
    return {'value':value if seen else None,'status':'RECORDED' if seen else 'UNKNOWN',
            'domain':domain if seen else None,'unit':unit if seen else None,'field':field if seen else None}


def header_value(fields):
    header = fields.get('header')
    t = header.get('stamp') if isinstance(header,dict) else None
    if isinstance(t,dict) and isinstance(t.get('sec'),int) and isinstance(t.get('nanosec'),int):
        if 0 <= t['nanosec'] < 1_000_000_000:
            return t['sec']*1_000_000_000+t['nanosec']
    return None


def clocks(fields, log, publish, manus=False, observation=False, native=False):
    out = {key:stamp(None) for key in CLOCKS}
    out['mcap_log'] = stamp(log,'collector_host_realtime' if native else 'mcap_record_clock',
                            'ns','mcap.log_time')
    out['mcap_publish'] = stamp(publish,'collector_host_realtime' if native else 'mcap_publish_clock',
                                'ns','mcap.publish_time')
    out['ros_header'] = stamp(header_value(fields),'message_ros_clock','ns','header.stamp')
    if manus:
        out['sdk_source'] = stamp(fields.get('source_timestamp_us'),'manus_sdk_compressed_timestamp',
                                  None,'source_timestamp_us')
        out['sdk_receive'] = stamp(fields.get('receive_timestamp_us'),
                                   'publisher_steady_clock' if native else 'sdk_receive_clock_unverified',
                                   'us','receive_timestamp_us')
    if observation:
        for target,key,domain in (
            ('collector_receive','receive_ros_timestamp_ns','collector_ros_clock'),
            ('collector_monotonic','host_observed_monotonic_ns','collector_host_monotonic'),
            ('collector_realtime','host_realtime_ns','collector_host_realtime'),
        ):
            out[target] = stamp(fields.get(key),domain,'ns',key)
    return out


def message_fields(msg):
    """Keep every decoded field. Non-finite values have explicit path and kind."""
    invalid = []

    def flatten(value, path):
        if isinstance(value,float) and not math.isfinite(value):
            invalid.append({'path':path,'value':str(value)})
            return None
        if value is None or isinstance(value,(bool,int,float,str)):
            return value
        if isinstance(value,(list,tuple,bytes,bytearray)):
            return [flatten(x,f'{path}/{i}') for i,x in enumerate(value)]
        slots = getattr(value,'__slots__',None)
        if slots is None:
            raise TypeError(f'Unhandled decoded field type: {type(value).__name__}, {path}')
        return {name:flatten(getattr(value,name),f'{path}/{name}') for name in slots}
    return flatten(msg,''), invalid


def xyz(msg, keys):
    if not isinstance(msg,dict):
        return None
    out = [msg.get(k) for k in keys]
    return out if all(isinstance(x,(float,int)) and not isinstance(x,bool) for x in out) else None


def manus_geometry(fields, expected_side):
    raw = fields.get('raw_nodes')
    issues = []
    if not isinstance(raw,list) or not raw:
        raw = []
        issues.append('MISSING_RAW_NODES')
    if len(raw)!=fields.get('raw_node_count'):
        issues.append('RAW_NODE_COUNT_MISMATCH')
    side = fields.get('side')
    if not side:
        issues.append('UNKNOWN_SIDE')
    elif str(side).strip().lower()!=expected_side:
        issues.append('SIDE_MISMATCH')
    nodes, identity = [], []
    for i,item in enumerate(raw):
        item = item if isinstance(item,dict) else {}
        pose = item.get('pose') or {}
        node = {k:item.get(k) for k in ('node_id','parent_node_id','joint_type','chain_type')}
        identity.append(list(node.values()))
        pos = xyz(pose.get('position'),'xyz')
        quat = xyz(pose.get('orientation'),'xyzw')
        nodes.append(dict(**node,original_order=i,node_position=pos,node_orientation=quat,
                          geometry_valid=bool(pos is not None and quat is not None and any(quat)),
                          measured_validity=item.get('validity')))
    ids = [n['node_id'] for n in nodes]
    if None in ids or len(set(ids))!=len(ids):
        issues.append('MISSING_OR_DUPLICATE_NODE_ID')
    if any(n['joint_type'] is None or n['chain_type'] is None or n['parent_node_id'] is None for n in nodes):
        issues.append('MISSING_TOPOLOGY_FIELDS')
    if any(not n['geometry_valid'] for n in nodes):
        issues.append('INVALID_NODE_GEOMETRY')
    header = fields.get('header') or {}
    return dict(node_count=len(nodes),nodes=nodes,original_node_order=list(range(len(nodes))),
                frame_id=header.get('frame_id') or None,raw_side=side,issues=issues,
                validity={'geometry':'INVALID' if issues else 'STRUCTURE_OK','sensor_status':'UNKNOWN'},
                topology_sha256=sha256(stable_json(identity).encode()).hexdigest())
