"""Read and preroll signed historical HMB geometry, without candidate-dependent sampling."""
from pathlib import Path
import sys

import numpy as np
from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY, HMBReader
from dexgym.tasks.direct.wholebody_s05.p02.intent import semantic_order
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, asset_files, digest
from run_study import E,E0,F,load,write
from workspace import prep


DOCS=Path('docs/s0-5/evidence')
HERE=DOCS/'P03_SIM0'
P02=DOCS/'P02'


def config():
    plan=load(HERE/'plan.json')
    base=load(P02/'plan.json')
    paths=[
        (E/'hmb_manifest.json',plan['hmb_manifest_sha256']),
        (F/'p02_handoff_manifest.json',base['handoff_sha256']),
        (P02/'final_handoff/execution_receipt.json',plan['p02_receipt_sha256']),
        (P02/'final_handoff/limited_scope_candidate.json',plan['p02_limited_candidate_sha256']),
        (P02/'refinement/fitted_candidates.json',plan['fits_sha256']),
        (Path(base['model']),plan['reviewed_model_sha256'])]
    for path,h in paths:
        if digest(path)!=h:
            raise ValueError(f'Signed source differs: {path}')
    def relative_assets(tree):
        return {path.split('/source/dexgym/assets/',1)[-1]:sha for path,sha in tree.items()}
    if relative_assets(asset_files(base['model']))!=relative_assets(load(P02/'refinement/model_assets_sha256.json')):
        raise ValueError('Signed asset tree differs')
    return plan,base,load(F/'p02_handoff_manifest.json'),load(E/'production_plan.json')


def readers(handoff):
    return {s:HMBReader(E/'hmb_manifest.json',E/'production_plan.json',E0/'candidate_config.json',
                handoff['ubuntu_products'],s,GEOMETRY) for s in ('Train','Validation')}


def topology(a,meta,codes):
    result=[]
    for s,side in enumerate(('left','right')):
        token=meta['dictionary']['topology_id'][int(codes[s])]
        result.append(next(v for v in meta['topologies'] if v['stream']==f'manus_{side}' and v['version']==token))
    return result


def geometry(a,meta,robot,cal,ok):
    n=len(ok)
    local=np.full((n,2,5,4,3),np.nan)
    human=np.full_like(local,np.nan)
    desired=np.full((n,2,4,3),np.nan)
    ids=a['topology_id'][:,:2]
    qualified=ok.copy()
    for code in np.unique(ids[ok],axis=0):
        ticks=np.flatnonzero(ok&(ids==code).all(axis=1))
        source=a['pose'][ticks,:2]
        order=list(map(semantic_order,topology(a,meta,code)))
        result=prep(source,order,robot,cal)
        whole=result['usable']
        qualified[ticks[~whole]]=False
        ix=ticks[whole]
        local[ix]=result['local'][whole]
        desired[ix]=result['desired'][whole]
        accepted=source[whole]
        human[ix]=np.stack([accepted[:,s,idx,:3]-accepted[:,s,r:r+1,:3][:,:,None,:]
                             for s,(r,idx) in enumerate(order)],axis=1)
    return qualified,local,human,desired


def slice_source(root,part,a,meta,frame,tag):
    start,end=frame
    t=np.arange(start,end)
    topo=a['topology_id'][t,:2]
    if not np.all(topo==topo[0]):
        raise ValueError('Selected window would cross a topology generation')
    path=root/(tag+'.npz')
    payload={
        'wrist':a['pose'][t,2:4,0],
        'raw_tracker':a['pose'][t,4:6,0],
        'palm':a['pose'][t,:2],
        'sdk_common':a['sdk_common_pose'][t,:2],
        'source_index':a['source_index'][t,:4,0],
        'tick_ns':a['tick_ns'][t],
        'joint_valid':a['joint_valid'][t],
        'safety_permission':a['safety_permission'][t]}
    np.savez_compressed(path,**payload)
    output=dict(partition=part['key'],root=part['parent'],split=part['split'],
                 start=start,end=end,topology=topology(a,meta,topo[0]),file=path.name,sha256=digest(path))
    write(root/(tag+'.json'),output)
    return output


def extract(mode,out):
    plan,base,handoff,prod=config()
    split={'train':'Train','validation':'Validation'}[mode]
    out=out/mode
    out.mkdir(mode=0o700)
    robot=RobotModel(base['model'],base['clearance_m'])
    cal=load(P02/'mapping_contract.json')
    reader=readers(handoff)[split]
    blocks=[]
    partitions=[]
    selected={}
    longest=None
    tasks={v['root']:v['task'] for v in prod['sources']}
    total=nvalid=0
    for index,p in enumerate([p for p in prod['partitions'] if p['split']==split]):
        a,meta,src,_=reader.read(p['key'])
        total+=len(a['joint_valid'])
        sim=np.asarray(a['joint_valid'],dtype=bool)
        nvalid+=int(sim.sum())
        ok,local,human,desired=geometry(a,meta,robot,cal,sim)
        n=len(ok)
        # Preserve immutable source table indices and diagnostic generations.
        blocks.append(dict(wrist=a['pose'][:,2:4,0],hmb_joint=sim,geom_valid=ok,
            local=local,human=human,desired=desired,
            tick_ns=a['tick_ns'],segment=a['segment_id'],part=np.full(n,index,dtype=np.int32),
            source_index=a['source_index'][:,:4,0],clock_quality=a['clock_quality'][:,:4],
            reason=a['reason_code'][:,:4],
            pose_valid=a['pose_valid'][:,:4],safety_permission=a['safety_permission']))
        task=tasks[p['parent']]
        for seg in sorted(set(a['segment_id'][ok]) - {-1}):
            indices=np.flatnonzero(ok & (a['segment_id']==seg))
            if not len(indices) or not np.all(np.diff(indices)==1):
                continue
            length=len(indices)
            lo=int(indices[0])
            if length>=plan['short_window_ticks'] and task not in selected:
                window=slice_source(out,p,a,meta,(lo,lo+plan['short_window_ticks']),task)
                selected[task]=window|{'part_id':index,'task':task}
            if longest is None or length>longest['length']:
                longest={'part_id':index,'start':lo,'end':lo+length,'length':length,'part':p,'task':task}
        partitions.append(dict(key=p['key'],root=p['parent'],task=task,count=n,index=index,
                              source_table_rows=len(src),calibration_ids=meta['dictionary']['calibration_id'],
                              global_start=total-n,identity=meta['identity']))
        if (index+1)%40==0:
            print(split,index+1,total,flush=True)
    expected=load(E/'full_coverage.json')[split]['counts']
    if (total,nvalid)!=(expected['planned_ticks'],expected['causal.joint_valid']):
        raise ValueError('Signed per-split denominators differ')
    if longest:
        lp=longest['part']
        a,meta,_,_=reader.read(lp['key'])
        longest=slice_source(out,lp,a,meta,(longest['start'],longest['end']),'longest')|longest
        longest.pop('part')
    np.savez_compressed(out/'full_geometry.npz',
                        **{k:np.concatenate([r[k] for r in blocks],axis=0) for k in blocks[0]})
    write(out/'index.json',dict(split=split,planned=total,hmb_joint_valid=nvalid,partitions=partitions,
        windows=selected,longest=longest,full_sha256=digest(out/'full_geometry.npz'),
        evidence_scope='Historical causal simulated availability. Synthetic events are added after extraction.'))
    print(split,total,nvalid,sorted(selected),None if longest is None else longest['length'],flush=True)


if __name__=='__main__':
    extract(sys.argv[1],Path(sys.argv[2]))
