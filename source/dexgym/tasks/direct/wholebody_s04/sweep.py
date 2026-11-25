"""Score each retained frame with S04-8000 using batched MuJoCo physics."""
import json
import time

import numpy as np
import torch

from dexgym.tasks.direct.wholebody_s0_1.bank import TaskBank
from dexgym.tasks.direct.wholebody_s04.task import Env
from dexgym.tasks.direct.wholebody_s04.hand_points import articulated_points

class ScanBank(TaskBank):
    same_frame=False

    def advance_frame(self,frame,steps,recorded=None,*,max_steps=None):
        if self.same_frame:
            return frame
        return super().advance_frame(frame,steps,recorded,max_steps=max_steps)


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False))
    tmp.replace(path)


def measure(actual, goal, depth):
    tips=goal.hand[...,-1,:]
    g=goal.wrist[:,:,None]+tips @ goal.rotation.transpose(-1,-2)
    actual_local=actual.hand[...,-1,:]
    pred=actual.wrist[:,:,None]+actual_local @ actual.rotation.transpose(-1,-2)
    arm=actual.wrist[:,:,None]+tips @ actual.rotation.transpose(-1,-2)
    mm=lambda x:1000*x.norm(dim=-1).amax(dim=-1)
    return torch.cat((mm(pred-g),mm(arm-g),mm(actual_local-tips),1000*depth[:,None]),dim=-1)


def windows(clips,length=208,overlap=26):
    return [(max(c['start'],a-overlap),a,min(c['stop'],a+length))
            for c in clips for a in range(c['start'],c['stop'],length)]


@torch.inference_mode()
def sweep(bank,model,sim,actor_weight,path,split,*,demo=False,collect_anchor=True):
    env=Env(sim,bank,model,49,evaluation=True)
    phy=env.direct
    phy.segment_steps=500
    phy.decoder.check_bank(bank.rsi_q)
    actor=env.task.make_actor().to(sim.device).eval()
    actor.load_state_dict(actor_weight)
    device=bank.rsi_q.device
    n=sim.num_envs
    if demo:
        source=bank.clips[17]
        assert source['id']=='181022_3348_3817', source
        offset=source['start']-3348
        job=[(offset+3679,offset+3704,offset+3817)]
        length=len(bank.rsi_q)
    else:
        job=sorted(windows(bank.clips),key=lambda x:x[2]-x[0],reverse=True)
        length=len(bank.rsi_q)
    rows=torch.full((length,7),float('nan'),dtype=torch.float16,device=device)
    flag=torch.zeros(length,dtype=torch.int8,device=device)
    contact_scan=torch.zeros(length,dtype=torch.float32,device=device)
    arm_metrics=torch.full((length,4),float("nan"),dtype=torch.float32,device=device)
    # Codes 1 scored, 2 physics stop, 3 reset transient after stop, 4 unsafe reset.
    start=time.monotonic()
    next_job=0
    current=np.zeros(n,dtype=np.int64)
    score_start=np.zeros(n,dtype=np.int64)
    stop=np.zeros(n,dtype=np.int64)
    cooldown=np.zeros(n,dtype=np.int8)
    active=np.zeros(n,dtype=bool)
    pending=np.ones(n,dtype=bool)
    visited=0
    num_steps=0
    dropped=0
    scored=0
    anchor_hist=[]
    anchor_action=[]
    if n>1:
        phy.bank.same_frame=True
        first_safe=bank.eligible[0].expand(n)
        phy.reset(frames=first_safe,exact=True)
        phy.bank.same_frame=False
    while pending.any() or active.any():
        reset_lanes=[]
        reset_frames=[]
        for lane in np.flatnonzero(pending):
            if not active[lane] or current[lane]>=stop[lane]:
                if next_job>=len(job):
                    active[lane]=False
                    continue
                current[lane],score_start[lane],stop[lane]=job[next_job]
                next_job+=1
                cooldown[lane]=0
                active[lane]=True
            reset_lanes.append(lane)
            reset_frames.append(current[lane])
        if not active.any():
            break
        if reset_lanes:
            ids=torch.tensor(reset_lanes,device=device)
            frames=torch.tensor(reset_frames,device=device)
            phy.bank.same_frame=True
            phy.reset(ids,frames=frames,exact=True)
            phy.bank.same_frame=False
        goal=phy.goal
        observation=phy.observation()
        action=actor(observation)
        phy.step(action)
        transition=phy.transition
        drop=transition['drop'].cpu().numpy() & active
        values=measure(phy.actual,goal,transition['depth']).to(torch.float16)
        lanes=torch.as_tensor(active,device=device)
        active_rows=torch.as_tensor(current[active],device=device)
        contact_scan.scatter_reduce_(0,active_rows,1000*transition['depth'][lanes],
                                     reduce='amax',include_self=True)
        keeps=active&(current>=score_start)
        masks=keeps & ~drop & (cooldown==0)
        chosen=torch.as_tensor(current[masks],device=device)
        rows[chosen]=values[torch.as_tensor(masks,device=device)]
        flag[chosen]=1
        outcome=transition["rollout_stats"]
        actual=articulated_points(phy.actual,phy.task.observe.proximal)
        desired=articulated_points(goal,phy.task.observe.proximal)
        extra=(1000.*(actual-desired).norm(dim=-1)).amax((1,2))
        values_arm=torch.stack((outcome["wrist_rotation_worst_deg"],
                               outcome["wrist_translation_worst_mm"],extra,
                               outcome["arm_residual_rate_deg_s"]),dim=1)
        arm_metrics[chosen]=values_arm[torch.as_tensor(masks,device=device)]
        if split=='train' and collect_anchor and num_steps%9==0:
            normal=torch.as_tensor(masks,device=device)&(values[:,:2].amax(-1)<12)&(values[:,6]<.5)
            count=normal.nonzero().flatten()
            if len(count):
                indices=count[torch.randperm(len(count),device=device)[:8]]
                anchor_hist.append(observation[indices].to(torch.float16).cpu().numpy())
                anchor_action.append(action[indices].to(torch.float16).cpu().numpy())
        flag[torch.as_tensor(current[keeps&drop],device=device)]=2
        flag[torch.as_tensor(current[keeps&~drop&(cooldown>0)],device=device)]=3
        visited+=int(keeps.sum())
        dropped+=int((keeps&drop).sum())
        scored+=int(masks.sum())
        cooldown=np.maximum(0,cooldown-1)
        cooldown[drop]=8
        current[active]+=1
        pending=active&((current>=stop)|drop)
        num_steps+=1
        if num_steps%80==0 or not active.any():
            secs=time.monotonic()-start
            stat=dict(split=split,phase='screening',visited=visited,scored=scored,target=sum(x[2]-x[1] for x in job),
                      dropped=dropped,clips_started=next_job,total_chunks=len(job),frames_per_s=visited/max(secs,.1),
                      seconds=secs)
            write(path/'sweep_progress.json',stat)
            print(json.dumps(stat),flush=True)
    assert visited==sum(x[2]-x[1] for x in job)
    if not demo:
        assert visited==length
    np.savez_compressed(path/f'sweep_{split}.npz',error_arm_shape_depth_mm=rows.cpu().numpy(),
                        status=flag.cpu().numpy(),max_contact_mm=contact_scan.cpu().numpy(),
                        wrist_and_hand=arm_metrics.cpu().numpy())
    if split=='train' and collect_anchor:
        if not anchor_hist:
            raise RuntimeError('Expected reference-policy histories from normal replay frames')
        np.savez_compressed(path/'retention.npz', observation=np.concatenate(anchor_hist),
                            action=np.concatenate(anchor_action))
    stat=dict(split=split,phase='screened',visited=visited,scored=scored,target=visited,dropped=dropped,
              seconds=time.monotonic()-start)
    write(path/f'sweep_{split}.json',stat)
    print(json.dumps(stat),flush=True)
