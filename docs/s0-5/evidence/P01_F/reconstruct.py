"""Reconstruct selected official HMB rows directly from original MCAP and independent math."""
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from mcap.reader import NonSeekingReader

from dexgym.tasks.direct.wholebody_s05.data.schema_registry import EmbeddedDecoders
from dexgym.tasks.direct.wholebody_s05.hmb_reader import (
    ACTOR_FIELDS, HMBReader, GEOMETRY, SIMULATED, DIAGNOSTIC, actor_features, eligible_history,
)


module=spec_from_file_location('p01f_archive',Path(__file__).with_name('scan_archive.py'))
arc=module_from_spec(module)
module.loader.exec_module(arc)
STREAMS=arc.STREAMS
KEYS=('node_id','parent_node_id','joint_type','chain_type')


def plain(x):
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if getattr(x,'__slots__',None):
        return {name:plain(getattr(x,name)) for name in x.__slots__}
    return x


def xyz_pose(data):
    return np.array([data['position'][a] for a in 'xyz']+[data['orientation'][a] for a in 'xyzw'])


def qnorm(q):
    return q/np.linalg.norm(q)


def qmul(a,b):
    av,bv=a[:3],b[:3]
    return np.r_[a[3]*bv+b[3]*av+np.cross(av,bv),a[3]*b[3]-av@bv]


def qinv(q):
    return qnorm(q)*[-1,-1,-1,1]


def rotate(q,xyz):
    q=qnorm(q)
    u=np.cross(q[:3],xyz)
    return xyz+2*(q[3]*u+np.cross(q[:3],u))


def compose(a,b):
    return np.r_[a[:3]+rotate(a[3:],b[:3]),qnorm(qmul(qnorm(a[3:]),qnorm(b[3:])))]


def angle(q,r):
    z=qmul(qinv(q),qnorm(r))
    return float(2*np.arctan2(np.linalg.norm(z[:3]),abs(z[3])))


def interp(a,b,w):
    q,r=qnorm(a[3:]),qnorm(b[3:])
    if q@r<0:
        r=-r
    d=np.clip(q@r,-1,1)
    phi=np.arccos(d)
    rot=qnorm((1-w)*q+w*r) if phi<1e-7 else qnorm(
        (np.sin((1-w)*phi)*q+np.sin(w*phi)*r)/np.sin(phi))
    return np.r_[(1-w)*a[:3]+w*b[:3],rot]


def header(fields):
    h=(fields.get('header') or {}).get('stamp')
    return None if h is None else 10**9*h['sec']+h['nanosec']


def raw_pose(fields,source,metadata):
    s=source['stream']
    if not s.startswith('manus_'):
        p=xyz_pose(fields['pose'])[None,:]
        return p,p,None
    raw=fields['raw_nodes']
    nodes=[{**{k:r[k] for k in KEYS},'original_order':j} for j,r in enumerate(raw)]
    top=next(t for t in metadata['topologies'] if (t['stream'],t['version'])==(s,source['topology_id']))
    arc.check(len(raw)==fields['raw_node_count']==25 and fields['side'].strip().lower()==s.split('_')[1] and
              nodes==top['joints'],'Raw topology or side differs')
    parents={q['node_id']:q['parent_node_id'] for q in nodes}
    root,= [i for i,p in parents.items() if i==p]
    for child in parents:
        seen=set()
        while child!=root:
            arc.check(child in parents and child not in seen,'Raw skeleton has orphan or cycle')
            seen.add(child)
            child=parents[child]
    p=np.stack([xyz_pose(q['pose']) for q in raw])
    origin=p[next(j for j,q in enumerate(nodes) if q['node_id']==root)]
    q=qinv(origin[3:])
    palm=np.array([np.r_[rotate(q,r[:3]-origin[:3]),qnorm(qmul(q,qnorm(r[3:])))] for r in p])
    return p,palm,nodes


def pointer(s):
    r=s['record']
    return r['part_id'],r['physical_message']


def required_state(actors,s,selected_index):
    a=s[selected_index]
    cap=a['capture_bounds_ns'][0]
    ref=a['identity']
    matches=[i for i in actors[a['stream']] if s[i]['raw_valid'] and s[i]['fresh'] and s[i]['source_eligible']
             and s[i]['identity']['instance']==ref['instance'] and s[i]['clock_generation']==a['clock_generation']
             and s[i]['calibration_id']==a['calibration_id'] and s[i]['topology_id']==a['topology_id']
             and s[i]['capture_bounds_ns'][1]<cap and s[i]['arrival_bounds_ns'][1]<a['arrival_bounds_ns'][0]]
    return max(matches,key=lambda i:(s[i]['capture_bounds_ns'][1],s[i]['identity']['writer_sequence']),default=None)


class Partition:
    def __init__(self, products, entry, cfg, goals):
        self.b=entry['partition']
        self.path=products/entry['evidence']['path']
        self.meta=arc.read(self.path/'metadata.json')
        self.sources=list(arc.rows(self.path/'sources.jsonl'))
        self.groups=defaultdict(list)
        for i,r in enumerate(self.sources):
            self.groups[r['stream']].append(i)
        fields=arc.read(arc.D/'hmb_schema.json')
        self.a={m:arc.arrays(self.path,m,fields,self.meta['ticks'],self.meta['N']) for m in ('causal','offline')}
        self.params=cfg['parameters']
        self.goals=goals
        self.requests=set()
        self._tracker_matches={}

    def bracket(self, mode, t, j):
        a=self.a[mode]
        return [int(i) for i in a['source_index'][t,j] if i>=0]

    def extract_requests(self):
        for g in self.goals:
            radius=4 if g['kind']=='extremum' else 0
            ticks=list(range(max(0,g['tick']-radius),min(self.meta['ticks'],g['tick']+radius+1)))
            if 'raw_row' in g and g['raw_row'] is not None:
                self.requests.add(g['raw_row'])
                s=self.sources[g['raw_row']]
                i=next((i for i in self.groups[s['stream']] if self.sources[i]['identity']['writer_sequence']==
                        s['identity']['writer_sequence']-1),None)
                if i is not None:
                    self.requests.add(i)
            for t in ticks:
                for mode,a in self.a.items():
                    safety=int(a['safety_source_index'][t])
                    if safety>=0:
                        self.requests.add(safety)
                    for j in range(len(STREAMS)):
                        indices=self.bracket(mode,t,j)
                        self.requests.update(indices)
                        if mode=='causal' and a['velocity_valid'][t,j] and indices:
                            previous=required_state(self.groups,self.sources,indices[0])
                            if previous is not None:
                                self.requests.add(previous)
                if g['kind']=='extremum':
                    ct=self.a['causal']['tick_ns'][t]
                    for stream, indices in self.groups.items():
                        # At most two raw stream neighbors per tick in this fixed review context.
                        closest=sorted(indices,key=lambda i:abs(self.sources[i]['capture_bounds_ns'][1]-ct))[:2]
                        self.requests.update(closest)

    def tracker_candidates(self,side,t):
        reg='tracker_'+side
        indices=self.bracket('causal',t,STREAMS.index(reg))
        if not indices:
            return []
        i=indices[0]
        mono=self.sources[i]['capture_bounds_ns'][0]
        candidates=sorted(self.groups[reg+'_raw'],key=lambda r:abs(self.sources[r]['capture_bounds_ns'][0]-mono))[:12]
        self.requests.update(candidates)
        self._tracker_matches[side,t]=(i,candidates)
        return candidates


def read_mcap(partitions,registry):
    requests=defaultdict(dict)
    for p in partitions:
        for i in p.requests:
            row=p.sources[i]
            part,k=pointer(row)
            r=row['record']
            previous=requests[part].get(k)
            arc.check(previous is None or previous==r,'Conflicting raw reference across aliquots')
            requests[part][k]=r
    wire=EmbeddedDecoders()
    decoded={}
    for part, asking in sorted(requests.items()):
        asset=Path(registry[part]['canonical_uri'])
        arc.check(arc.digest(asset)==registry[part]['file_sha256'],'Raw asset checksum differs')
        physical=set(asking).union(r['observation_message'] for r in asking.values())
        found={}
        with asset.open('rb') as f:
            for index,x in enumerate(NonSeekingReader(f,validate_crcs=True,record_size_limit=32*2**20).iter_messages(log_time_order=False)):
                if index in physical:
                    found[index]=x
                    if found.keys()==physical:
                        break
        arc.check(found.keys()==physical,f'Missing raw physical indices in {part}')
        for k,r in asking.items():
            s,c,d=found[k]
            os,oc,od=found[r['observation_message']]
            for data,h in [(s.data,r['schema_sha256']),(d.data,r['cdr_sha256']),
                           (os.data,r['observation_schema_sha256']),(od.data,r['observation_cdr_sha256'])]:
                arc.check(sha256(data).hexdigest()==h,'Physical CRC/CDR/schema witness differs')
            arc.check(c.topic==r['topic'] and r['observation_message']==k+1,'Raw role or pairing differs')
            m=plain(wire.decode(c.message_encoding,s,d.data))
            o=plain(wire.decode(oc.message_encoding,os,od.data))
            arc.check(o['stream_id']==r['stream'] and o['source_instance_id']==r['source_instance'] and
                      o['source_sequence_id']==r['writer_sequence'] and o['host_realtime_ns']==d.log_time==od.log_time,
                      'Observation role/instance/sequence/time mismatch')
            decoded[part,k]=(m,o)
    return decoded


def compare_pose(a,b,tol):
    t=float(np.linalg.norm(a[:,:3]-b[:,:3],axis=1).max())
    r=max(angle(q,z) for q,z in zip(a[:,3:],b[:,3:],strict=True))
    arc.check(t<=tol['translation_m'] and r<=tol['rotation_rad'],f'Pose residual {t:.3g} m / {r:.3g} rad')
    return [t,r]


def numeric_record(p,mode,t,j,decoded,origin,tol):
    a=p.a[mode]
    sources=p.sources
    ids=p.bracket(mode,t,j)
    result=dict(mode=mode,tick=t,stream=STREAMS[j],pose_valid=bool(a['pose_valid'][t,j]),
                velocity_valid=bool(a['velocity_valid'][t,j]),reason=int(a['reason_code'][t,j]),
                safety_permission=bool(a['safety_permission'][t]),
                source=[sources[i]['record'] for i in ids])
    if not a['pose_valid'][t,j]:
        arc.check(not a['velocity_valid'][t,j] and not a['node_mask'][t,j].any(),
                  'Invalid pose is eligible or fabricated')
        return result
    capture=[]
    views=[]
    for i in ids:
        source=sources[i]
        raw,obs=decoded[pointer(source)]
        ct=obs['host_observed_monotonic_ns']-origin
        arc.check(source['raw_clock']['collector_realtime']['value']==obs['host_realtime_ns'] and
                  source['raw_clock']['collector_monotonic']['value']==obs['host_observed_monotonic_ns'] and
                  source['capture_bounds_ns']==[ct,ct] and
                  source['arrival_bounds_ns']==[ct+8_000_000,ct+8_000_000],
                  'Physical collector clock or simulated bounds differ')
        record_fields=raw.get('header') or {}
        arc.check(source['recorded_frame']==record_fields.get('frame_id'),'Recorded frame mismatch')
        if j<2:
            arc.check(raw['source_timestamp_us']==source['identity']['sdk_timestamp_raw'] and
                      raw['receive_timestamp_us']==source['identity']['sdk_receive_raw'],'SDK identity differs')
        if j in (2,3):
            arc.check(raw['valid'] is True,'Invalid registered tracker was used')
        views.append(raw_pose(raw,source,p.meta))
        capture.append(ct)
    nodes=a['node_mask'][t,j]
    result['version']=dict(calibration=sources[ids[0]]['calibration_id'],topology=sources[ids[0]]['topology_id'],
                           clock=sources[ids[0]]['clock_generation'])
    if mode=='causal':
        ct,= capture
        tick,cut=a['tick_ns'][t],a['cutoff_ns'][t]
        arc.check(ct<=tick and ct+8_000_000<=cut and a['age_ns'][t,j]==tick-ct and
                  a['capture_bounds_ns'][t,j].tolist()==[ct,ct] and
                  a['arrival_bounds_ns'][t,j].tolist()==[ct+8_000_000,ct+8_000_000],'Actor availability contract differs')
        fresh=[i for i in p.groups[STREAMS[j]] if sources[i]['raw_valid'] and sources[i]['fresh']
               and sources[i]['source_eligible'] and sources[i]['capture_bounds_ns'][1]<=tick and
               sources[i]['arrival_bounds_ns'][1]<=cut]
        latest=max(fresh,key=lambda i:(sources[i]['arrival_bounds_ns'][1],sources[i]['identity']['writer_sequence']))
        arc.check(latest==ids[0],'Causal row is not the latest fresh available source')
        target=views[0][1]
        raw_sdk=views[0][0]
        result['age_ns']=int(tick-ct)
        if a['velocity_valid'][t,j]:
            previous=required_state(p.groups,sources,ids[0])
            arc.check(previous is not None,'No predecessor for valid velocity')
            prv=sources[previous]
            r,ob=decoded[pointer(prv)]
            dt=ct-(ob['host_observed_monotonic_ns']-origin)
            p0=raw_pose(r,prv,p.meta)[1]
            arc.check(0<dt<=p.params['age_ns'] and a['velocity_age_ns'][t,j]==tick-ct,
                      'Velocity predecessor timing differs')
            velocity=(target[:,:3]-p0[:,:3])/(dt*1e-9)
            result['velocity_previous']=prv['record']
            result['velocity_dt_ns']=dt
    else:
        lo,hi=capture
        w=0. if hi==lo else (a['tick_ns'][t]-lo)/(hi-lo)
        arc.check(lo<=a['tick_ns'][t]<=hi and a['bracket_ns'][t,j].tolist()==capture and
                  abs(w-a['interpolation_weight'][t,j])<=1e-14 and a['bracket_safe'][t,j],'Offline physical bracket differs')
        target=np.stack([interp(l,r,w) for l,r in zip(views[0][1],views[1][1],strict=True)])
        raw_sdk=np.stack([interp(l,r,w) for l,r in zip(views[0][0],views[1][0],strict=True)])
        arc.check(bool(a['velocity_valid'][t,j])==(hi>lo),'Independent offline derivative mask differs')
        if hi>lo:
            velocity=(views[1][1][:,:3]-views[0][1][:,:3])/((hi-lo)*1e-9)
            result['velocity_dt_ns']=hi-lo
        result.update(bracket_ns=capture,weight=w)
    result['pose_residual_m_rad']=compare_pose(target,a['pose'][t,j,nodes],tol)
    if j<2:
        result['sdk_residual_m_rad']=compare_pose(raw_sdk,a['sdk_common_pose'][t,j,nodes],tol)
    if a['velocity_valid'][t,j]:
        actual=a['velocity_m_s'][t,j,nodes]
        arc.check(np.allclose(velocity,actual,atol=tol['velocity_abs_m_s'],rtol=tol['velocity_relative']),
                  f'Independent velocity mismatch: {float(np.max(np.abs(velocity-actual))):.6g} m/s')
        result['velocity_max_abs_residual']=float(np.max(np.abs(velocity-actual)))
    return result


def calibration_check(p,side,t,decoded,cal_file,limits):
    reg,candidates=p._tracker_matches[side,t]
    b,bo=decoded[pointer(p.sources[reg])]
    key=header(b)
    options=[(i,decoded[pointer(p.sources[i])][0]) for i in candidates]
    matched=[(i,a) for i,a in options if header(a)==key]
    arc.check(key is not None and len(matched)==1,'Raw/registered header pairing is missing or ambiguous')
    i,a=matched[0]
    conf=yaml.safe_load(cal_file.read_bytes())['tracker_calibration_node']['ros__parameters']
    role=side+'_wrist'
    arc.check(conf[role+'_parent']==role+'_raw' and conf[role+'_child']==role,'Calibration side differs')
    transform=np.r_[conf[role+'_translation'],conf[role+'_quaternion_xyzw']]
    x=compose(xyz_pose(a['pose']),transform)
    src,dst=a['header']['frame_id'],b['header']['frame_id']
    if src!=dst:
        arc.check((src,dst)==('openvr_world','teleop_world'),'Unapproved tracker frame transform')
        x=compose(np.r_[0,0,0,0,2**-.5,2**-.5,0],x)
    y=xyz_pose(b['pose'])
    linear=float(np.linalg.norm(x[:3]-y[:3]))
    angular=angle(x[3:],y[3:])
    arc.check(linear<=limits['m'] and angular<=limits['rad'],'Independently reconstructed registration exceeds approved bound')
    return dict(side=side,header_ns=key,raw=p.sources[i]['record'],registered=p.sources[reg]['record'],
                translation_m=linear,rotation_rad=angular,calibration_sha256=arc.digest(cal_file),frames=[src,dst])


def verify_safety(p,mode,t,decoded,origin):
    a=p.a[mode]
    i=int(a['safety_source_index'][t])
    if i<0:
        arc.check(not a['safety_permission'][t],'No evidence for safety permission')
        return None
    row=p.sources[i]
    raw,obs=decoded[pointer(row)]
    ct=obs['host_observed_monotonic_ns']-origin
    arc.check(row['safe'] is raw.get('safe') and row['stream']=='tracker_safety','Safety payload differs')
    if a['safety_permission'][t]:
        arc.check(raw['safe'] is True and 0<=a['tick_ns'][t]-ct<=p.params['safety_age_ns']
                  and (mode!='causal' or ct+8_000_000<=a['cutoff_ns'][t]),'Safety freshness or availability differs')
    return dict(index=row['record'],safe=raw['safe'],permitted=bool(a['safety_permission'][t]),capture_ns=ct)


def context(p,goal,decoded):
    t=goal['tick']
    a=p.a['causal']
    lo,hi=max(0,t-4),min(len(a['tick_ns']),t+5)
    timestamps=a['tick_ns'][lo:hi]
    source=[]
    for i in sorted(p.requests):
        s=p.sources[i]; ct=s['capture_bounds_ns'][0]
        if ct<timestamps[0]-60_000_000 or ct>timestamps[-1]+60_000_000:
            continue
        f,_=decoded[pointer(s)]
        pose=f.get('pose')
        if s['stream'].startswith('manus') and s['raw_valid']:
            index=min(goal.get('node',4),24)
            node=f['raw_nodes'][index]
            pose=node['pose']|{'node_id':node['node_id']}
        source.append(dict(stream=s['stream'],capture_ns=ct,record=s['record'],identity=s['identity'],
                           valid=s['raw_valid'],fresh=s['fresh'],safe=s['safe'],calibration=s['calibration_id'],
                           topology=s['topology_id'],clock=s['clock_generation'],
                           position_orientation=pose))
    changes=[]
    for stream in STREAMS:
        trace=sorted((r for r in source if r['stream']==stream and r['position_orientation'] is not None),
                     key=lambda r:r['capture_ns'])
        for x,y in zip(trace,trace[1:]):
            b,c=(xyz_pose(r['position_orientation']) for r in (x,y))
            if (not np.isfinite(b).all() or not np.isfinite(c).all() or
                    np.linalg.norm(b[3:])==0 or np.linalg.norm(c[3:])==0):
                continue
            changes.append(dict(stream=stream,from_record=x['record']['physical_message'],
                to_record=y['record']['physical_message'],dt_ns=y['capture_ns']-x['capture_ns'],
                displacement_m=float(np.linalg.norm(c[:3]-b[:3])),rotation_rad=angle(b[3:],c[3:])))
    return dict(tick_indices=list(range(lo,hi)),ticks_ns=timestamps.tolist(),joint=a['joint_valid'][lo:hi].tolist(),
                safety=a['safety_permission'][lo:hi].tolist(),
                reason=a['reason_code'][lo:hi].tolist(),causal_age_ns=a['age_ns'][lo:hi].tolist(),
                adjacent_raw=source,raw_pose_changes=changes)


def execute(e1,old_a,archive,out):
    manifest,plan,_=arc.identity()
    config=arc.read(arc.E0/'candidate_config.json')
    d_plan=arc.read(arc.D/'frozen_config.json')['plan']
    policy=arc.read(arc.F/'sample_plan.json')
    tol=policy['tolerance']
    choices=arc.read(archive/'selected_samples.json')
    products=e1/'products'
    by_key={r['partition']['key']:r for r in arc.read(products/'build_index.json')['partitions']}
    reg={r['id']:r for r in arc.rows(old_a/'registry.jsonl')}
    goals=defaultdict(list)
    for root,r in choices['roots'].items():
        goals[r['partition']].append(dict(**r,kind='root',name=root))
    for kind in ('scenarios','top'):
        for name,listings in choices[kind].items():
            for r in listings:
                goals[r['partition']].append(dict(**r,kind='scenario' if kind=='scenarios' else 'extremum',name=name))
    # Bounded decoding by physical indices. Load source tables for selected partitions only.
    partitions=[Partition(products,by_key[k],config,v) for k,v in sorted(goals.items())]
    for p in partitions:
        p.extract_requests()
        for g in p.goals:
            if g['kind']=='root':
                for side in ('left','right'):
                    p.tracker_candidates(side,g['tick'])
    wire=read_mcap(partitions,reg)
    result=[]; anomalies=[]; failures=[]; counts=Counter()
    roots_confirmed=set()
    reused={(x['record']['part_id'],x['record']['physical_message'])
            for v in arc.read(arc.E/'raw_traceability.json')['cases'].values() for x in v['matched']}
    for p in partitions:
        row=reg[p.b['recording_id']]
        raw_file=Path(row['canonical_uri'])/'manifest.yaml'
        origin=yaml.safe_load(raw_file.read_bytes())['start_monotonic_ns']
        candidates=[c for c in row['calibration_refs'] if c['sha256']==p.meta['identity']['calibration_sha256']]
        cal_file=next(Path(c['path']) for c in candidates if 'path' in c)
        arc.check(arc.digest(cal_file)==p.meta['identity']['calibration_sha256'],'Calibration file bytes differ')
        for g in p.goals:
            t=g['tick']
            evidence=dict(goal=g, partition=p.b, tick_ns=int(p.a['causal']['tick_ns'][t]), checks=[],calibration=[],safety={})
            initial_failures=len(failures)
            for mode in ('causal','offline'):
                try:
                    evidence['safety'][mode]=verify_safety(p,mode,t,wire,origin)
                    counts['safety_checks']+=1
                except (ValueError,KeyError) as ex:
                    failures.append(dict(goal=g,mode=mode,stream='safety',error=str(ex)))
                for j,s in enumerate(STREAMS):
                    try:
                        report=numeric_record(p,mode,t,j,wire,origin,tol)
                        evidence['checks'].append(report)
                        counts['arrays_pose_checked']+=int(report['pose_valid'])
                        counts['arrays_velocity_checked']+=int(report['velocity_valid'])
                        counts['invalid_mask_checked']+=int(not report['pose_valid'])
                    except (ValueError,KeyError) as ex:
                        failures.append(dict(goal=g,mode=mode,stream=s,error=str(ex)))
            if g['kind']=='root':
                for side in ('left','right'):
                    try:
                        evidence['calibration'].append(calibration_check(p,side,t,wire,cal_file,d_plan['calibration_error_limits']))
                        counts['tracker_calibration_pairs']+=1
                    except (ValueError,KeyError) as ex:
                        failures.append(dict(goal=g,stream=side,error=str(ex)))
                if len(failures)==initial_failures:
                    roots_confirmed.add(g['name'])
            if g['name']=='DUPLICATE_SOURCE':
                src=p.sources[g['raw_row']]
                previous=next(i for i in p.groups[src['stream']] if p.sources[i]['identity']['writer_sequence']==
                              src['identity']['writer_sequence']-1)
                old,now=(wire[pointer(z)][0]['raw_nodes'] for z in (p.sources[previous],src))
                evidence['repeated_sdk']=dict(equal_raw_nodes=old==now,older=p.sources[previous]['record'],
                    newer=src['record'],old_sdk=p.sources[previous]['identity'],new_sdk=src['identity'])
            if g['kind']=='extremum':
                established=len(failures)==initial_failures
                elapsed=[dict(mode=r['mode'],stream=r['stream'],dt_ns=r.get('velocity_dt_ns'),source=r['source'],
                              previous=r.get('velocity_previous')) for r in evidence['checks'] if 'velocity_dt_ns' in r]
                window=context(p,g,wire)
                anomalies.append(dict(goal=g,binding=p.b,classification='RECORDED_CHANGE_WITH_PROXY_TIME'
                    if established else 'INDEPENDENT_RECONSTRUCTION_DISAGREEMENT',
                    physical_cause='UNCONFIRMED', tracking_interruption_observed=not all(window['joint']),
                    source_interval_checks=elapsed,window=window,
                    disposition='RETAIN_FROZEN_QC' if established else 'HOLD_FOR_REVIEW'))
            result.append(evidence)
    arc.write(out/'raw_reconstruction.json',dict(counts=counts,checked_goals=len(result),cases=result,
              raw_pairs=len(wire),raw_files=len({k[0] for k in wire}),reuse_vs_e1=len(reused&set(wire)),
              confirmed_roots=sorted(roots_confirmed),missing_roots=sorted(set(choices['roots'])-roots_confirmed),
              missing_scenarios=choices['missing_scenarios'],failures=failures,tolerances=tol))
    arc.write(out/'tail_anomalies.json',dict(top_k=policy['top_k'],cases=anomalies,verified_physical_timing=False))
    actual_reader=check_reader(out,products,choices,by_key,reg)
    plot_receipt=visualize(out,partitions,wire,choices)
    arc.write(out/'visual_receipt.json',plot_receipt)
    print(json.dumps(dict(roots=len(roots_confirmed),goals=len(result),counts=counts,raw_pairs=len(wire),
                          extremes=len(anomalies),failures=len(failures),reader=actual_reader,plots=plot_receipt)))
    return not failures and len(roots_confirmed)==len(plan['sources']) and not choices['missing_scenarios']


def visualize(out,partitions,wire,choices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selected=next(p for p in partitions if any(g['kind']=='root' and p.b['split']=='Train' for g in p.goals))
    t=next(g['tick'] for g in selected.goals if g['kind']=='root')
    fig=plt.figure(figsize=(10,6))
    for j in (0,1):
        a=selected.a['causal']
        src=selected.sources[selected.bracket('causal',t,j)[0]]
        _,palm,top=raw_pose(wire[pointer(src)][0],src,selected.meta)
        xyz=a['sdk_common_pose'][t,j,:25,:3]
        for mode,points in enumerate((xyz,palm)):
            ax=fig.add_subplot(2,2,j*2+mode+1,projection='3d')
            points=points[:,:3]
            ix={r['node_id']:i for i,r in enumerate(top)}
            for r in top:
                u,v=ix[r['parent_node_id']],ix[r['node_id']]
                ax.plot(*points[[u,v]].T,c=('teal','darkorange')[j],lw=1.3)
            ax.scatter(*points.T,s=8)
            ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',
                   title=f"{STREAMS[j]} / {['SDK common','palm'][mode]} (25 nodes)")
    fig.suptitle(f"{selected.b['parent']} · {selected.b['label']} · tick {t} / {a['tick_ns'][t]} ns · SIMULATED")
    fig.tight_layout()
    fig.savefig(out/'manus_geometry.png',dpi=155)
    plt.close(fig)
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection='3d')
    for j,side,color in ((2,'left','teal'),(3,'right','darkorange')):
        for mode,style in [('causal','-'),('offline',':')]:
            a=selected.a[mode]
            ax.plot(*a['pose'][:,j,0,:3].T,style,c=color,label=f'{side} / {mode}')
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.legend()
    grid=selected.a['causal']['tick_ns']
    fig.suptitle(f"{selected.b['parent']} / {selected.b['label']} · {grid[0]}–{grid[-1]} ns\nSIMULATED / OFFLINE_PROXY")
    fig.tight_layout()
    fig.savefig(out/'wrist_trajectories.png',dpi=155)
    plt.close(fig)
    cases=[('normal_motion','normal_motion'),('bimanual_motion','normal_bimanual'),('DUPLICATE_SOURCE','repeat_and_age'),
           ('UNSAFE','unsafe_and_recovery'),('recovery','recovery_transition'),('right_original_tip','tip_error_tail'),
           ('causal.tracker_right.speed','wrist_speed_tail'),('STALE','stale_age_boundary')]
    generated=['manus_geometry.png','wrist_trajectories.png']
    for name,label in cases:
        all_goals=[(p,g) for p in partitions for g in p.goals if g['name']==name]
        if name=='bimanual_motion':
            speed_limit=arc.read(arc.D/'frozen_config.json')['plan']['dynamic_threshold_m_s']
            for q in partitions:
                if not any(g['kind']=='root' for g in q.goals):
                    continue
                a=q.a['causal']
                speeds=np.linalg.norm(np.where(a['node_mask'][:,:4,:,None],
                                      a['velocity_m_s'][:,:4],0.),axis=-1)
                moving=np.any(speeds>speed_limit,axis=2)&a['velocity_valid'][:,:4]
                t=np.flatnonzero(a['joint_valid']&(moving[:,0]|moving[:,2])&(moving[:,1]|moving[:,3]))
                if len(t):
                    all_goals=[(q,{'tick':int(t[len(t)//2])})]
                    break
        if not all_goals:
            continue
        p,g=max(all_goals,key=lambda x:x[1].get('value',0))
        i=g['tick']
        ca,of=p.a['causal'],p.a['offline']
        lo,hi=max(0,i-15),min(len(ca['tick_ns']),i+16)
        x=(ca['tick_ns'][lo:hi]-ca['tick_ns'][i])/1e6
        fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for j,side,color in [(0,'left','teal'),(1,'right','darkorange')]:
            for data,style,mode in [(ca,'-','simulated'),(of,':','offline diagnostic')]:
                p0=data['pose'][lo:hi,j+2,0,:3]
                axs[0].plot(x,np.linalg.norm(p0,axis=1),style,color=color,label=f'{side} wrist / {mode}')
            if not name.endswith('.speed') and not name.endswith('_original_tip'):
                hand=ca['pose'][lo:hi,j,int(min(g.get('node',4),24)),:3]
                axs[1].plot(x,np.linalg.norm(hand,axis=1)*1000,color=color,label=f'{side} palm node radius')
            axs[2].plot(x,np.where(ca['pose_valid'][lo:hi,j],ca['age_ns'][lo:hi,j]/1e6,np.nan),color=color,
                        label=f'{side} age (ms)')
        ax=axs[2].twinx()
        ax.step(x,ca['safety_permission'][lo:hi],where='post',color='purple',lw=1,alpha=.8)
        ax.step(x,ca['joint_valid'][lo:hi],where='post',color='gray',alpha=.65)
        ax.set(ylim=(-.05,1.05),ylabel='purple safety / gray joint')
        for ax in axs:
            ax.axvline(0,c='black',alpha=.25)
            ax.legend(loc='upper right',fontsize=8)
            ax.grid(alpha=.2)
        if name=='DUPLICATE_SOURCE' and g.get('raw_row') is not None:
            x_dup=(p.sources[g['raw_row']]['capture_bounds_ns'][0]-ca['tick_ns'][i])/1e6
            axs[2].axvline(x_dup,c='red',linestyle='--',label='recorded duplicate SDK identity')
            axs[2].legend(loc='upper right',fontsize=8)
        axs[0].set_ylabel('wrist position norm (m)')
        if name.endswith('.speed'):
            j=STREAMS.index(name.split('.')[1])
            for data,label,style in ((ca,'simulated','-'),(of,'offline diagnostic',':')):
                axs[1].plot(x,np.linalg.norm(data['velocity_m_s'][lo:hi,j,g['node']],axis=1),style,label=label)
            axs[1].set_ylabel('selected node speed (m/s)')
            axs[1].legend(loc='upper right',fontsize=8)
        elif name.endswith('_original_tip'):
            j=0 if name.startswith('left') else 1
            diff=ca['pose'][lo:hi,j,g['node'],:3]-of['pose'][lo:hi,j,g['node'],:3]
            axs[1].plot(x,np.linalg.norm(diff,axis=1)*1000,color='darkorange')
            axs[1].set_ylabel('alignment difference (mm)')
        else:
            axs[1].set_ylabel('palm-local (mm)')
        axs[2].set_ylabel('age (ms)')
        axs[2].set_xlabel('relative to selected control tick (ms)')
        fig.suptitle(f"{name} · {p.b['parent']} · {p.b['label']} · {ca['tick_ns'][i]} ns\nSIMULATED / OFFLINE_PROXY")
        fig.tight_layout()
        path=label+'.png'
        fig.savefig(out/path,dpi=150)
        plt.close(fig)
        generated.append(path)
    return generated


def check_reader(out,products,choices,by_key,reg):
    args=(arc.E/'hmb_manifest.json',arc.E/'production_plan.json',arc.E0/'candidate_config.json',products)
    raw_assets={k:r['canonical_uri'] for k,r in reg.items() if 'canonical_uri' in r}
    results=[]
    for split in ('Train','Validation'):
        key=next(x['partition'] for x in choices['roots'].values() if by_key[x['partition']]['partition']['split']==split)
        other=next(k for k,p in by_key.items() if p['partition']['split']!=split)
        for purpose in (GEOMETRY,SIMULATED,DIAGNOSTIC):
            reader=HMBReader(*args,split,purpose)
            data,meta,sources,frames=reader.read(key,raw_assets=raw_assets)
            arc.check(other not in reader.keys() and key in reader.keys(),'Wrong root split admitted by reader')
            try:
                reader.read(other)
            except KeyError:
                pass
            else:
                raise ValueError('Reader permits cross-split source')
            proof=dict(split=split,purpose=purpose,partition=key,number_of_indexed_partitions=len(reader.keys()),
                       source_rows=len(sources),frames=frames)
            if purpose==SIMULATED:
                t=next(i for i in range(meta['ticks']) if eligible_history(data,i))
                before=reader.history(key,t)
                after_mutation={k:v.copy() for k,v in data.items()}
                for name,value in after_mutation.items():
                    value[t+1:]=0
                    if name not in ACTOR_FIELDS:
                        value[:]=0
                truncated={k:v[:t+1] for k,v in data.items()}
                for prefix in (after_mutation,truncated):
                    actual=actor_features(prefix,t)
                    arc.check(tuple(actual)==ACTOR_FIELDS,'Unexpected online feature')
                    for k in actual:
                        np.testing.assert_array_equal(before[k],actual[k])
                arc.check(not eligible_history(data,3),'Reader crossed the partition start')
                boundary={k:v.copy() for k,v in data.items()}
                boundary['segment_id'][t-2]=-1
                arc.check(not eligible_history(boundary,t),'Reader crossed a same-partition invalid segment')
                proof.update(tick=t,prefix_equal=True,feature_names=ACTOR_FIELDS)
            results.append(proof)
    arc.write(out/'reader_check.json',dict(cases=results,status='PASS'))
    return len(results)


if __name__=='__main__':
    if not execute(*(Path(x).resolve() for x in sys.argv[1:])):
        sys.exit(3)
