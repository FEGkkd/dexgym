"""Bounded offline interpolation and deployment-reusable causal event consumption."""
from bisect import bisect_right
from dataclasses import dataclass
import heapq
import numpy as np

from .clock_domains import qualify
from .geometry import interpolate
from .sample_identity import Identity,IdentityHistory


STREAMS=('manus_left','manus_right','tracker_left','tracker_right','tracker_left_raw','tracker_right_raw')
REASONS=('VALID','HELD','STALE','DROPOUT','DUPLICATE_SOURCE','OUT_OF_ORDER','CLOCK_JUMP',
         'UNSYNC','UNSAFE','SAFETY_UNKNOWN','CALIBRATION_INVALID','SOURCE_UNKNOWN','TRACKER_INVALID',
         'SOURCE_RESTART','SOURCE_IMMUTABILITY_CONFLICT','GAP','VERSION_BOUNDARY','MANUS_INVALID')


def invalid_reason(stream):
    return 'MANUS_INVALID' if stream.startswith('manus_') else 'TRACKER_INVALID'


@dataclass(frozen=True)
class Event:
    stream: str
    identity: Identity
    ref: dict
    known_ns: int
    capture: tuple | None
    arrival: tuple | None
    quality: str
    pose: np.ndarray | None = None
    valid: bool = True
    safe: bool | None = None
    version: tuple = ()
    calibrated: bool = True
    segment: str = 'BODY'
    capture_proof: str | None = None
    arrival_proof: str | None = None

    def qualified(self):
        return qualify(self.capture,self.arrival,self.quality,self.capture_proof,self.arrival_proof)


def batch_causal(events,ticks,params,quality):
    core=CausalSynchronizer(params,quality)
    for e in events:
        core.offer(e)
    return [core.step(tick,cutoff) for tick,cutoff in ticks]


class CausalSynchronizer:
    """Queue visible emissions independently from their capture and arrival bounds."""
    def __init__(self,params,quality='VERIFIED'):
        if quality not in {'VERIFIED','PROXY','SIMULATED'}:
            raise ValueError('A synchronizer requires an explicit usable time domain')
        self.params=params
        self.quality=quality
        self.arrival=[]
        self.capture=[]
        self.counter=0
        self.current={}
        self.velocity={}
        self.histories={}
        self.last_tick=self.last_cutoff=None

    def offer(self,event):
        valid=event.qualified()==self.quality
        delivered=max(event.known_ns,event.arrival[1]) if valid else event.known_ns
        heapq.heappush(self.arrival,(delivered,self.counter,event,valid))
        self.counter+=1

    def _apply(self,event,quality_ok,updated,reasons):
        s=event.stream
        bound=event.capture
        prev=self.current.get(s)
        if not quality_ok:
            reasons[s]='UNSYNC'
            return
        # Initial-state data are usable only if bound to original times and independent proofs.
        if event.segment!='BODY' and (event.segment!='INITIAL_STATE' or self.quality!='VERIFIED'):
            reasons[s]='SOURCE_UNKNOWN'
            return
        if prev and prev.identity.instance==event.identity.instance and bound[1]<prev.capture[1]:
            reasons[s]='OUT_OF_ORDER'
            return
        hist=self.histories.setdefault(s,IdentityHistory())
        status=hist.observe(event.identity)
        if status in {'DUPLICATE_SOURCE','OUT_OF_ORDER','SOURCE_UNKNOWN','SOURCE_IMMUTABILITY_CONFLICT'}:
            reasons[s]=status
            return
        self.velocity.pop(s,None)
        if (prev and status=='NEW' and prev.valid and event.valid and
                prev.version==event.version and prev.pose is not None and event.pose is not None and
                prev.calibrated and event.calibrated):
            elapsed=(bound[1]-prev.capture[1])*1e-9
            if elapsed>0:
                self.velocity[s]=(event.pose[:,:3]-prev.pose[:,:3])/elapsed
        if status in {'CLOCK_JUMP','SOURCE_RESTART'}:
            reasons[s]=status
            self.current.pop(s,None)
            return
        self.current[s]=event
        updated.add(s)
        reasons[s]='VALID'

    def step(self,tick_ns,cutoff_ns):
        if self.last_tick is not None and (tick_ns<self.last_tick or cutoff_ns<self.last_cutoff):
            raise ValueError('Past control ticks are immutable')
        while self.arrival and self.arrival[0][0]<=cutoff_ns:
            delivery,seq,e,q=heapq.heappop(self.arrival)
            capture=e.capture[1] if q else tick_ns
            heapq.heappush(self.capture,(capture,seq,delivery,e,q))
        events=[]
        while self.capture and self.capture[0][0]<=tick_ns:
            _,seq,delivery,e,q=heapq.heappop(self.capture)
            events.append((delivery,seq,e,q))
        updated,reasons=set(),{}
        for _,_,event,quality_ok in sorted(events):
            self._apply(event,quality_ok,updated,reasons)
        self.last_tick,self.last_cutoff=tick_ns,cutoff_ns
        safety=self.current.get('tracker_safety')
        safety_reason=None
        if not safety or safety.safe is None or tick_ns-safety.capture[0]>self.params['safety_age_ns']:
            safety_reason='SAFETY_UNKNOWN'
        elif safety.safe is False or not safety.valid:
            safety_reason='UNSAFE'
        out={}
        for s in STREAMS:
            item=self.current.get(s)
            if item is None:
                why=reasons.get(s,'DROPOUT')
                out[s]=dict(valid=False,reason=why,ref=None,pose=None,velocity=None,age_ns=None,
                            capture=None,arrival=None,input_reason=why,version=None)
                continue
            age=tick_ns-item.capture[0]
            why=(safety_reason or ('CALIBRATION_INVALID' if not item.calibrated else None) or
                 (invalid_reason(s) if not item.valid else None) or ('STALE' if age>self.params['age_ns'] else None) or
                 ('VALID' if s in updated else 'HELD'))
            valid=why in {'VALID','HELD'}
            out[s]=dict(valid=valid,reason=why,ref=item.ref,pose=item.pose if valid else None,
                        velocity=self.velocity.get(s) if valid else None,age_ns=age,
                        capture=item.capture,arrival=item.arrival,input_reason=reasons.get(s),
                        version=item.version)
        joint,spans=joint_mask(out,self.params)
        return dict(tick_ns=tick_ns,cutoff_ns=cutoff_ns,streams=out,joint_valid=joint,
                    joint_reason='VALID' if joint else 'UNSYNC',skew_ns=spans,quality=self.quality,
                    safety_ref=safety.ref if safety else None)


def joint_mask(out,params):
    pairs=(('manus_left','manus_right'),('tracker_left','tracker_right'),
           ('manus_left','tracker_left'),('manus_right','tracker_right'))
    span={}
    usable=True
    for index,(a,b) in enumerate(pairs):
        limit=params.get('left_right_skew_ns' if index<2 else 'wrist_hand_skew_ns',params['skew_ns'])
        if out[a]['valid'] and out[b]['valid']:
            l,r=out[a]['capture'],out[b]['capture']
            delta=max(l[1],r[1])-min(l[0],r[0])
            span[a+'__'+b]=delta
            usable=usable and delta<=limit
        else:
            span[a+'__'+b]=None
            usable=False
    return usable,span


class OfflineAlignment:
    """Segmented interpolation in an explicit offline domain."""
    def __init__(self,events,params):
        self.params=params
        self.events={s:[] for s in (*STREAMS,'tracker_safety')}
        self.times={}
        history={}
        generation=dict.fromkeys(self.events,0)
        for event in sorted(events,key=lambda e:(e.known_ns,e.ref.get('physical_message',0))):
            if event.segment!='BODY' or event.qualified()=='UNUSABLE':
                continue
            hist=history.setdefault(event.stream,IdentityHistory())
            reason=hist.observe(event.identity)
            if reason in {'OUT_OF_ORDER','DUPLICATE_SOURCE','SOURCE_UNKNOWN','SOURCE_IMMUTABILITY_CONFLICT'}:
                continue
            if reason in {'CLOCK_JUMP','SOURCE_RESTART'}:
                generation[event.stream]+=1
            self.events[event.stream].append((event,reason,generation[event.stream]))
        for s,items in self.events.items():
            items.sort(key=lambda x:x[0].capture[1])
            self.times[s]=[e.capture[1] for e,_,_ in items]

    def _safe(self,lo,hi):
        items=self.events['tracker_safety']; times=self.times['tracker_safety']
        k=bisect_right(times,lo)-1
        if k<0:
            return 'SAFETY_UNKNOWN'
        last=times[k]
        generation=items[k][2]
        for index in range(k,bisect_right(times,hi)):
            event,reason,epoch=items[index]
            if generation!=epoch or reason in {'CLOCK_JUMP','SOURCE_RESTART'}:
                return 'SAFETY_UNKNOWN'
            if times[index]-last>self.params['safety_age_ns']:
                return 'SAFETY_UNKNOWN'
            if event.safe is not True or not event.valid:
                return 'UNSAFE' if event.safe is False else 'SAFETY_UNKNOWN'
            last=times[index]
        return 'SAFETY_UNKNOWN' if hi-last>self.params['safety_age_ns'] else None

    def _bracket(self,s,tick):
        times,items=self.times[s],self.events[s]
        k=bisect_right(times,tick)-1
        if k<0 or (times[k]!=tick and k+1>=len(times)):
            return None,'DROPOUT'
        j=k if times[k]==tick else k+1
        a,ar,ae=items[k]; b,br,be=items[j]
        for reason in (ar,br):
            if reason in {'CLOCK_JUMP','SOURCE_RESTART'}:
                return None,reason
        if not a.calibrated or not b.calibrated:
            return None,'CALIBRATION_INVALID'
        if not a.valid or not b.valid or a.pose is None or b.pose is None:
            return None,invalid_reason(s)
        if ae!=be or a.identity.instance!=b.identity.instance or a.version!=b.version or a.pose.shape!=b.pose.shape:
            return None,'VERSION_BOUNDARY'
        gap=times[j]-times[k]
        if gap>self.params['gap_ns']:
            return None,'GAP'
        safety=self._safe(times[k],times[j])
        if safety:
            return None,safety
        w=0 if not gap else (tick-times[k])/gap
        p=a.pose if not gap else interpolate(a.pose,b.pose,w)
        velocity=None if not gap else (b.pose[:,:3]-a.pose[:,:3])/(gap*1e-9)
        return dict(valid=True,reason='VALID',pose=p,velocity=velocity,refs=[a.ref,b.ref],weight=w,
                    time_interval=(times[k],times[j]),capture=(tick,tick),version=a.version),None

    def step(self,tick_ns):
        out={}
        for s in STREAMS:
            result,why=self._bracket(s,tick_ns)
            out[s]=result or dict(valid=False,reason=why,pose=None,velocity=None,refs=None,
                                  weight=None,time_interval=None,capture=None,version=None)
        valid,spans=joint_mask(out,self.params)
        return dict(tick_ns=tick_ns,streams=out,joint_valid=valid,skew_ns=spans,quality='OFFLINE_PROXY')
