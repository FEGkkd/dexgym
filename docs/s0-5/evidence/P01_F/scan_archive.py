"""Read-only corpus accounting and preregistered sample selection."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, GEOMETRY, SIMULATED, DIAGNOSTIC
from dexgym.tasks.direct.wholebody_s05.data.p01e_lock import verify_lock


F=Path('docs/s0-5/evidence/P01_F')
E=F.parent/'P01_E_E1'
E0=F.parent/'P01_E_E0'
D=F.parent/'P01_D/products'
STREAMS=('manus_left','manus_right','tracker_left','tracker_right','tracker_left_raw','tracker_right_raw')
SCOPE='E1_PRODUCTION'
SCHEMA='s05.hmb.simulation.v1'
ALLOWED='HISTORICAL_SIMULATION_DEVELOPMENT_ONLY'


def check(condition, why):
    if not condition:
        raise ValueError(why)


def digest(p):
    with p.open('rb') as f:
        return sha256_file(f)


def sha256_file(f):
    h=sha256()
    for chunk in iter(lambda: f.read(4*2**20), b''):
        h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def rows(p):
    return (json.loads(l) for l in p.open())


def write(p,o):
    with p.open('x',encoding='utf-8') as f:
        json.dump(o,f,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def identity():
    files={
        'hmb_manifest.json':'e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074',
        'production_plan.json':'0f94b842e2558ec04da257aff94d469ae9b965968fff1a4593382fc2eea09404',
        'review_receipt.json':'0930e732e92e6213742e8b48cd1921d980c1315c7c108bda1ea24c0c59de483b'}
    for name,h in files.items():
        check(digest(E/name)==h,f'Approved E1 identity differs: {name}')
    receipt=read(E/'review_receipt.json')
    for name,h in receipt['review_artifacts_sha256'].items():
        check(digest(E/name)==h,f'Versioned E1 report differs: {name}')
    for name,want in [('candidate_config.json',receipt['execution_candidate_sha256']),
                      ('execution_lock.json',receipt['execution_lock_sha256'])]:
        check(digest(E0/name)==want,name)
    lock=read(E0/'execution_lock.json')
    for path,want in lock['source_sha256'].items():
        check(digest(Path(path))==want,f'Approved production implementation differs: {path}')
    return read(E/'hmb_manifest.json'),read(E/'production_plan.json'),receipt


def arrays(path,mode,schema,ticks,n):
    fields=schema['arrays'][mode]
    dims={'T':ticks,'S':6,'N':n,'2':2,'3':3,'7':7}
    with np.load(path/(mode+'.npz'),allow_pickle=False) as f:
        check(set(f.files)==set(fields),f'Fields differ: {path}')
        a={}
        for key,(typ,shape,_,_) in fields.items():
            value=f[key]
            check(value.dtype==np.dtype(typ).newbyteorder('<') and
                  value.shape==tuple(dims[k] for k in shape.split(',')),f'Bad {mode}/{key}: {path}')
            a[key]=value
    return a


def check_arrays(a, mode, meta, sources, schema):
    p,v,n=a['pose_valid'],a['velocity_valid'],a['node_mask']
    perm=a['safety_permission']
    check(np.array_equal(n.any(axis=-1),p) and not (v&~p).any() and not (p&~perm[:,None]).any(),
          'Pose or velocity validity contradicts safety')
    check(np.isfinite(a['pose'][n]).all() and np.isfinite(a['velocity_m_s'][n&v[...,None]]).all(),
          'Nonfinite values at valid node')
    qual=schema['vocabularies']['quality'].index('SIMULATED' if mode=='causal' else 'OFFLINE_PROXY')
    check(np.array_equal(a['clock_quality'],np.where(p,qual,0)) and
          np.array_equal(a['velocity_quality'],np.where(v,qual,0)),'Unexpected historic clock quality')
    ids=a['source_index']
    check(np.all((ids>=-1)&(ids<len(sources))) and np.all(ids[...,0][p]>=0),'Bad raw source indices')
    for j,stream in enumerate(STREAMS):
        relevant=np.unique(ids[:,j][p[:,j]])
        check(all(sources[ix]['stream']==stream and sources[ix]['source_eligible']
                  and sources[ix]['raw_valid'] and sources[ix]['fresh'] for ix in relevant if ix>=0),
                  f'{stream} uses invalid, ineligible or stale identity')
    bound=a['capture_bounds_ns'] if mode=='causal' else a['bracket_ns']
    lo,hi=bound[...,0],bound[...,1]
    i=np.broadcast_to(a['tick_ns'][:,None],p.shape)
    cfg=meta['identity']['parameters']
    if mode=='causal':
        cut=a['arrival_bounds_ns'][...,1]
        check(np.all(hi[p]<=i[p]) and np.all(cut[p]<=np.broadcast_to(a['cutoff_ns'][:,None],p.shape)[p]) and
              np.array_equal(a['age_ns'][p],(i-lo)[p]) and np.all(a['age_ns'][p]<=cfg['age_ns']) and
              np.array_equal(a['velocity_age_ns'][v],a['age_ns'][v]),'Causal ages/bounds differ')
    else:
        check(np.all((lo<=i)[p]) and np.all((i<=hi)[p]) and np.all(ids[...,1][p]>=0) and
              np.all(a['bracket_safe'][p]) and np.all(hi[p]-lo[p]<=cfg['gap_ns']),'Unsafe offline brackets')
    joint=p[:,:4].all(axis=1)&perm
    for k,l,name in [(0,1,'left_right_skew_ns'),(2,3,'left_right_skew_ns'),
                     (0,2,'wrist_hand_skew_ns'),(1,3,'wrist_hand_skew_ns')]:
        joint&=(np.maximum(hi[:,k],hi[:,l])-np.minimum(lo[:,k],lo[:,l])<=cfg[name])
    check(np.array_equal(joint,a['joint_valid']),'Independent joint mask differs')
    sid=a['segment_id']
    expected=np.full(len(sid),-1,dtype=np.int32)
    for idx,(lo,hi) in enumerate(meta['segments'][mode]):
        check(hi-lo>=5 and np.all(sid[lo:hi]==idx) and joint[lo:hi].all() and
              not any(lo<x<hi for x in meta['diagnostic_non_actor']['boundary_cut_indices']),'Segment crosses a barrier')
        for field in ('clock_generation','calibration_id','topology_id'):
            check(np.all(a[field][lo:hi,:4]==a[field][lo,:4]),f'Segment crosses {field}')
        expected[lo:hi]=idx
    check(np.array_equal(sid,expected),'Segment reference differs')


def window_end(mask,n):
    count=np.r_[0,np.cumsum(mask,dtype=np.int64)]
    return count[n:]-count[:-n]==n


def window_stats(c,o,meta,policy):
    result={}
    for mode,a in [('causal',c),('offline',o)]:
        mask=a['joint_valid']
        spans=meta['segments'][mode]
        result.update({mode+'_joint':int(mask.sum()),mode+'_segments':len(spans),
                       mode+'_segment_ticks':sum(hi-lo for lo,hi in spans),
                       mode+'_safety_permission':int(a['safety_permission'].sum())})
        for j,s in enumerate(STREAMS):
            result[mode+'_pose_'+s]=int(a['pose_valid'][:,j].sum())
            result[mode+'_velocity_'+s]=int(a['velocity_valid'][:,j].sum())
        for width in policy['reference_ticks']:
            result[f'{mode}_reference_{width}_windows']=sum(max(0,hi-lo-width+1) for lo,hi in spans)
            result[f'{mode}_reference_{width}_segments']=sum(hi-lo>=width for lo,hi in spans)
    common=c['joint_valid']&o['joint_valid']
    result['common_tick']=int(common.sum())
    result['history_pose']=result['history_velocity']=result['history_common']=0
    for lo,hi in meta['segments']['causal']:
        s=slice(lo,hi)
        mask=c['joint_valid'][s]&c['pose_valid'][s,:4].all(axis=1)&c['safety_permission'][s]
        result['history_pose']+=int(window_end(mask,5).sum())
        result['history_velocity']+=int(window_end(mask&c['velocity_valid'][s,:4].all(axis=1),5).sum())
    for lo,hi in meta['segments']['offline']:
        for left,right in meta['segments']['causal']:
            size=min(hi,right)-max(lo,left)
            result['history_common']+=max(0,size-4)
    return result


def activity(o,meta,speed_threshold,dwell):
    v=o['velocity_valid'][:,:4]
    vmask=o['node_mask'][:,:4]&v[...,None]
    node=np.linalg.norm(np.where(vmask[...,None],o['velocity_m_s'][:,:4],0),axis=-1)
    moving=(node.max(axis=2)>speed_threshold)&v
    eligible=v.all(axis=1)
    left=moving[:,0]|moving[:,2]
    right=moving[:,1]|moving[:,3]
    def count(a):
        return int(np.count_nonzero(a&eligible))
    output=dict(static=count(~(left|right)),dynamic=count(left|right),activity_unknown=int((~eligible).sum()),
                left_only=count(left&~right),right_only=count(right&~left),both_sides=count(left&right),
                both_hands=count(moving[:,0]&moving[:,1]),both_wrist_translation=count(moving[:,2]&moving[:,3]))
    angle=2*np.arccos(np.clip(np.abs(np.sum(o['pose'][1:,2:4,0,3:]*o['pose'][:-1,2:4,0,3:],axis=-1)),0,1))
    same=(o['segment_id'][1:]==o['segment_id'][:-1])&(o['segment_id'][1:]>=0)
    spin=(angle>.002)&same[:,None]
    output.update(wrist_rotation_left=int(spin[:,0].sum()),wrist_rotation_right=int(spin[:,1].sum()),
                  both_wrist_rotation=int(spin.all(axis=1).sum()),
                  safety_release=int(np.count_nonzero(~o['safety_permission'][:-1]&o['safety_permission'][1:])),
                  joint_recovery=int(np.count_nonzero(~o['joint_valid'][:-1]&o['joint_valid'][1:])))
    output.update(start=0,stop=0)
    for lo,hi in meta['segments']['offline']:
        for t in range(lo+dwell,hi-dwell+1):
            a,b=slice(t-dwell,t),slice(t,t+dwell)
            if not eligible[t-dwell:t+dwell].all():
                continue
            any_motion=left|right
            output['start']+=int(not any_motion[a].any() and any_motion[b].all())
            output['stop']+=int(any_motion[a].all() and not any_motion[b].any())
    return output


class Selection:
    def __init__(self,policy):
        self.policy=policy
        self.roots={}
        self.cases=defaultdict(list)
        self.top=defaultdict(list)

    def rank(self,*s):
        return sha256(':'.join(map(str,(self.policy['seed'],*s))).encode()).hexdigest()

    def take(self,key,items,n,score):
        return sorted(key+items,key=score)[:n]

    def scenario(self,name,part,ticks,stream=None,raw=None):
        items=[dict(partition=part,tick=int(t),stream=stream,raw_row=raw) for t in ticks]
        self.cases[name]=self.take(self.cases[name],items,self.policy['scenario_cap'],
            lambda a:self.rank(name,a['partition'],a['tick'],a['stream']))

    def peak(self,name,part,values):
        values=np.asarray(values).reshape(len(values),-1)
        flat=values.flatten()
        valid=np.flatnonzero(np.isfinite(flat))
        ranked=valid[np.argsort(-flat[valid],kind='stable')[:self.policy['top_k']]]
        items=[dict(partition=part,tick=int(i//values.shape[1]),node=int(i%values.shape[1]),value=float(flat[i]))
               for i in ranked]
        self.top[name]=self.take(self.top[name],items,self.policy['top_k'],
            lambda x:(-x['value'],x['partition'],x['tick'],x['node']))

    def observe(self,b,c,o,meta,sources,schema):
        part=b['key']
        good=np.flatnonzero(c['joint_valid']&c['pose_valid'].all(axis=1)&o['pose_valid'].all(axis=1))
        if len(good):
            i=int(min(good,key=lambda j:abs(int(j)-len(c['tick_ns'])//2)))
            rank=(b['tail'],self.rank(part))
            current=self.roots.get(b['parent'])
            if current is None or rank<tuple(current['rank']):
                self.roots[b['parent']]=dict(partition=part,tick=i,rank=rank)
            self.scenario('normal_motion',part,[i])
            if b['tail']:
                self.scenario('natural_tail',part,[i])
        for label in ('DROPOUT','STALE','UNSAFE'):
            code=schema['vocabularies']['reason'].index(label)
            self.scenario(label,part,np.flatnonzero((c['reason_code']==code).any(axis=1)))
        self.scenario('recovery',part,1+np.flatnonzero(~c['joint_valid'][:-1]&c['joint_valid'][1:]))
        self.scenario('partition_boundary',part,[0,len(c['tick_ns'])-1])
        prev={}
        for idx,r in enumerate(sources):
            k=r['stream']
            if not k.startswith('manus'):
                continue
            old=prev.get(k)
            prev[k]=r['identity']
            i=r['identity']
            if (old and i['instance']==old['instance'] and i['writer_sequence']==old['writer_sequence']+1 and
                    i['sdk_timestamp_raw'] is not None and i['sdk_receive_raw'] is not None and
                    (i['sdk_timestamp_raw'],i['sdk_receive_raw'])==(old['sdk_timestamp_raw'],old['sdk_receive_raw'])):
                capture=r['capture_bounds_ns'][1]
                if not c['tick_ns'][0]<=capture<=c['tick_ns'][-1]:
                    continue
                tick=int(np.searchsorted(c['tick_ns'],capture))
                self.scenario('DUPLICATE_SOURCE',part,[tick],k,idx)
        for mode,a in [('causal',c),('offline',o)]:
            for stream in self.policy['extreme_speed_streams']:
                j=STREAMS.index(stream)
                valid=a['node_mask'][:,j]&a['velocity_valid'][:,j,None]
                speed=np.linalg.norm(a['velocity_m_s'][:,j],axis=-1)
                self.peak(f'{mode}.{stream}.speed',part,np.where(valid,speed,np.nan))
        for j,side in enumerate(('left','right')):
            top=next(t for t in meta['topologies'] if t['stream']==STREAMS[j])
            tips=[n for n,joint in enumerate(top['joints']) if joint['joint_type'].upper()=='TIP']
            distance=np.linalg.norm(c['pose'][:,j,:,:3]-o['pose'][:,j,:,:3],axis=-1)
            mask=np.zeros(distance.shape[1],dtype=bool)
            mask[tips]=True
            self.peak(side+'_original_tip',part,np.where(mask,distance,np.nan))
        both=(c['pose_valid'][:,2:4]&o['pose_valid'][:,2:4]).all(axis=1)
        rel=np.full(len(both),np.nan)
        if both.any():
            l,r=(a['pose'][both,2:4,0] for a in (c,o))
            relatives=[Rotation.from_quat(p[:,0,3:]).inv().apply(p[:,1,:3]-p[:,0,:3]) for p in (l,r)]
            rel[both]=np.linalg.norm(relatives[0]-relatives[1],axis=1)
        self.peak('relative_registered_wrists',part,rel)


def verify_ledger(index,plan,log):
    current={}
    edges={None:{'PLANNED'},'PLANNED':{'RUNNING','PAUSED'},'RUNNING':{'FAILED','PAUSED','RESUMED','PUBLISHED','QUARANTINED'},
           'FAILED':{'PAUSED','RESUMED'},'PAUSED':{'RESUMED'},'RESUMED':{'RUNNING','PAUSED'}}
    by_key={p['key']:p for p in plan['partitions']}
    for i,line in enumerate(rows(log)):
        k=line['key']
        status=current[k]['status'] if k in current else None
        check(line['scope']==SCOPE and line['ordinal']==i and line['partition']==by_key[k] and
              line['status'] in edges.get(status,set()),f'Invalid E1 ledger event {i}')
        current[k]=line
    check(set(current)==set(by_key),'Ledger misses planned partitions')
    check(all(current[row['partition']['key']]['status']==row['status'] and
              current[row['partition']['key']]['evidence']==row['evidence'] for row in index['partitions']),
              'Ledger and final index differ')


def scan(e1,old_a,output):
    manifest,plan,receipt=identity()
    verify_lock(old_a,E0/'candidate_config.json',E0/'execution_lock.json')
    policy=read(F/'sample_plan.json')
    schema=read(D/'hmb_schema.json')
    cfg=read(E0/'candidate_config.json')
    assets=e1/'products'
    recorded=read(E/'output_sha256.json')
    signed=recorded['files']
    check(digest(old_a/'registry.jsonl')==cfg['registry_sha256'],'A1 Registry identity differs')
    a_sidecar=read(old_a/'output_sha256.json')
    for name in ('lineage.jsonl','prior_use.jsonl'):
        check(digest(old_a/name)==a_sidecar[name],f'A1 evidence differs: {name}')
    index=read(assets/'build_index.json')
    check(digest(assets/'build_index.json')==manifest['index_sha256'] and
          digest(assets/'build_log.jsonl')==signed['products/build_log.jsonl'] and
          digest(e1/'production_plan.json')==manifest['original_plan_sha256'],'E1 products plan/index/ledger hashes differ')
    check(index['scope']==plan['scope']==manifest['scope']==SCOPE and
          [row['partition'] for row in index['partitions']]==plan['partitions'],'Mixed archive or incomplete plan')
    verify_ledger(index,plan,assets/'build_log.jsonl')
    records={r['id']:r for r in rows(old_a/'registry.jsonl')}
    source_map={r['recording_id']:r for r in plan['sources']}
    for producer in ('source_whitelist.json','split.json','hmb_schema.json'):
        check(digest(D/producer)==cfg['artifact_sha256'][producer],f'D1 approved sidecar differs: {producer}')
    whitelist={r['recording_id']:r for r in read(D/'source_whitelist.json')['sources']}
    roots={role:set(r['root_episode_id'] for r in read(D/'split.json')[role]) for role in ('Train','Validation')}
    lineage=Counter()
    use=Counter()
    for r in rows(old_a/'lineage.jsonl'):
        lineage[str(r.get('relationship','UNKNOWN'))] += 1
    for r in rows(old_a/'prior_use.jsonl'):
        use[str(r.get('usage_role',r.get('use',r.get('role','UNKNOWN'))))] += 1
    check(len(source_map)==len(whitelist)==32 and not roots['Train']&roots['Validation'] and
          set(source_map)==set(whitelist),'Bad source whitelist or split overlap')
    raw_parts={}
    root_data={}
    for key,r in source_map.items():
        base=records[key]
        w=whitelist[key]
        check(base['root_episode_id']==r['root']==w['root_episode_id'] and base['producer_kind']=='LIVE_TELEOP' and
              base['status']=='ADMITTED_CANDIDATE' and r['root'] in roots[r['split']] and r['split']==w['split'] and
              r['interval_ns']==base['recorded_interval_ns']==w['recorded_interval_ns'] and
              r['manifest_sha256']==w['manifest_sha256'],'Non-Teleop or mismatched root admitted')
        # Metadata qualification and original raw identities are separate evidence.
        folder=Path(base['canonical_uri'])
        check(digest(folder/'manifest.yaml')==w['manifest_sha256'] and
              digest(folder/'run_snapshot.json')==w['snapshot_sha256'],'Raw provenance manifest differs')
        for group in w['assets'].values():
            for q in group:
                a=records[q['id']]
                check(a['file_sha256']==q['sha256'] and digest(Path(a['canonical_uri']))==q['sha256'],
                      f'Frozen MCAP SHA differs: {a["id"]}')
                raw_parts[q['id']]=q['sha256']
        root_data[r['root']]=dict(**r,session=base.get('session_id') or 'UNKNOWN',
                operator='UNKNOWN',calibration=w['calibration_sha256'],source_kind=base['producer_kind'])
    tick_grid={p['key']:(p['stop_ns']-p['start_ns']+19_999_999)//20_000_000 for p in plan['partitions']}
    by_group=defaultdict(Counter)
    partition_counters=[]
    selection=Selection(policy)
    verified={}
    check(digest(D/'frozen_config.json')==cfg['parent_d_config_sha256'],'D1 configuration differs')
    d_plan=read(D/'frozen_config.json')['plan']
    speed_threshold=d_plan['dynamic_threshold_m_s']
    for n,row in enumerate(index['partitions'],1):
        b=row['partition']
        check(row['status']=='PUBLISHED' and b['planned_ticks']==tick_grid[b['key']],f'Incomplete partition: {b["key"]}')
        rel=b['split']+'/'+b['parent'].replace(':','_')+'/'+b['label']
        path=(assets/rel).resolve()
        check(path.is_relative_to(assets.resolve()) and row['evidence']['path']==rel,'Archive path mismatch')
        for name in ('manifest.json','metadata.json','sources.jsonl','causal.npz','offline.npz'):
            item=f'products/{rel}/{name}'
            h=digest(e1/item)
            check(h==signed[item],f'Changed product: {item}')
            verified[item]=h
        part=read(path/'manifest.json')
        meta=read(path/'metadata.json')
        identity_record=meta['identity']
        r=source_map[b['recording_id']]
        check(meta['schema']==part['schema']==SCHEMA and meta['allowed_use']==ALLOWED and
              meta['sync_source_sha256']==manifest['sync_sha256'] and
              identity_record['publication_scope']==SCOPE and identity_record['execution_candidate_sha256']==receipt['execution_candidate_sha256'] and
              identity_record['verification']=='SIMULATED' and identity_record['parameters']==cfg['parameters'] and
              identity_record['root_episode_id']==b['parent']==r['root'] and identity_record['development_split']==r['split'] and
              identity_record['source_manifest_sha256']==r['manifest_sha256'] and
              [b['start_ns'],b['stop_ns']]==identity_record['raw_interval_realtime_ns'],'Foreign data blended into production')
        source=list(rows(path/'sources.jsonl'))
        check(all(raw_parts[x['record']['part_id']]==x['record']['asset_sha256'] for x in source),'Bad physical source')
        c,o=(arrays(path,m,schema,b['planned_ticks'],meta['N']) for m in ('causal','offline'))
        check(np.array_equal(c['tick_ns'],o['tick_ns']) and np.all(np.diff(c['tick_ns'])==20_000_000),'Partition tick mismatch')
        for m,a in [('causal',c),('offline',o)]:
            check_arrays(a,m,meta,source,schema)
        counts=window_stats(c,o,meta,policy)|activity(o,meta,speed_threshold,d_plan['stop_dwell_ticks'])
        counts.update(ticks=b['planned_ticks'],partitions=1,duration_ns=b['stop_ns']-b['start_ns'])
        for key in ('All',b['split'],'root:'+b['parent'],'task:'+str(r['task']), 'split_task:'+b['split']+':'+str(r['task']),
                    'date:'+r['capture_date'],'config:'+r['recorder_config_sha256']):
            by_group[key].update(counts)
        partition_counters.append(dict(key=b['key'],root=b['parent'],split=b['split'],counts=counts))
        selection.observe(b,c,o,meta,source,schema)
        if n%32==0:
            print(f'Checked {n}/{len(index["partitions"])} partitions',flush=True)
    check(set(verified)=={str(p.relative_to(e1)) for p in assets.glob('*/*/[0-9]*/*') if p.is_file()},
          'Archive contains extra/untracked partition files')
    expected=dict(ticks=116613,partitions=308,causal_joint=114532,offline_joint=109965,causal_segments=884,
                  offline_segments=2061,causal_segment_ticks=114332,offline_segment_ticks=108504)
    check(all(by_group['All'][k]==value for k,value in expected.items()) and len(verified)==1540,
          'Independent reference quantities differ')
    for role in roots:
        check(by_group[role]['ticks']==plan['source_totals'][role]['planned_ticks'],'Split totals differ')
    chosen=dict(roots=selection.roots,scenarios=selection.cases,top=selection.top,
                missing_roots=sorted(set(root_data)-set(selection.roots)),
                missing_scenarios=sorted(s for s in policy['scenarios'] if not selection.cases[s]))
    write(output/'independent_integrity.json',dict(status='PASS',hashes=verified,raw_hashes=raw_parts,
          historical_usage_events=sum(use.values()),historical_usage_roles=use,lineage_rows=sum(lineage.values()),
          source_count=len(root_data),split_overlap=0,archive_bytes=sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()),
          locked_source_files=len(read(E0/'execution_lock.json')['source_sha256'])))
    write(output/'history_reference_statistics.json',dict(groups=by_group,partitions=partition_counters,
          reference_ticks=policy['reference_ticks'],pose_history_ticks=5,speed_threshold_m_s=speed_threshold,
          wrist_angular_reference_rad_s=.1,rotation_scope='same valid offline segment thresholds; diagnostic only',
          roots=root_data,independent_capture_only=True))
    write(output/'selected_samples.json',chosen)
    print(json.dumps(dict(groups={k:by_group[k] for k in ('All','Train','Validation')},
                          missing_scenarios=chosen['missing_scenarios'],missing_roots=chosen['missing_roots'])))


if __name__=='__main__':
    scan(*(Path(a).resolve() for a in sys.argv[1:]))
