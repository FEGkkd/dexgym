"""Bounded offline interpolation and deployment-reusable causal event consumption."""
from bisect import bisect_right
from dataclasses import dataclass
import heapq
import numpy as np

from .clock_domains import qualify
from .geometry import interpolate
from .sample_identity import Identity,IdentityHistory


STREAMS=('manus_left','manus_right','tracker_left','tracker_right','tracker_left_raw','tracker_right_raw')
CORE_STREAMS=STREAMS[:4]
SAFETY='tracker_safety'
DISCONTINUITIES={'CLOCK_JUMP','CLOCK_GENERATION','SOURCE_RESTART','SOURCE_IMMUTABILITY_CONFLICT'}
REASONS=('VALID','HELD','STALE','DROPOUT','DUPLICATE_SOURCE','OUT_OF_ORDER','CLOCK_JUMP',
         'UNSYNC','UNSAFE','SAFETY_UNKNOWN','CALIBRATION_INVALID','SOURCE_UNKNOWN','TRACKER_INVALID',
         'SOURCE_RESTART','SOURCE_IMMUTABILITY_CONFLICT','GAP','VERSION_BOUNDARY','MANUS_INVALID',
         'CLOCK_GENERATION','RECOVERY_FRESHNESS','FRESHNESS_UNKNOWN')


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
    authenticated: bool = False
    clock_generation: str = 'proxy-v1'
    fresh: bool = True

    def qualified(self):
        return qualify(self.capture,self.arrival,self.quality,self.capture_proof,self.arrival_proof)


def event_key(event):
    identity=event.identity
    return (event.stream!=SAFETY,event.stream,identity.instance,identity.sequence,
            event.safe is True,event.safe is False,event.valid,event.calibrated,event.fresh,
            identity.fingerprint,identity.source_stamp,identity.receive_stamp,
            event.clock_generation,event.capture or (),event.arrival or (),event.ref.get('physical_message',-1))


def record_signature(event):
    return (event.identity,event.valid,event.fresh,event.calibrated,event.safe,event.pose is not None,
            event.version,event.clock_generation,event.quality,event.capture)


def batch_causal(events,ticks,params,quality):
    core=CausalSynchronizer(params,quality)
    for e in events:
        core.offer(e)
    return [core.step(tick,cutoff) for tick,cutoff in ticks]


class CausalSynchronizer:
    """Consume pose and interlock events in a single monotonic replay domain."""
    def __init__(self, params, quality='VERIFIED'):
        if quality not in {'VERIFIED', 'PROXY', 'SIMULATED'}:
            raise ValueError('A synchronizer requires an explicit usable time domain')
        self.params = params
        self.quality = quality
        self.pending = []
        self.capture = []
        self.counter = 0
        self.current = {}
        self.velocity = {}
        self.velocity_reason = {}
        self.blocked = {}
        self.histories = {}
        self.latest_records = {}
        self.generations = {}
        self.retired = set()
        self.latest_bounds = {}
        self.permission = None
        self.permission_reason = 'SAFETY_UNKNOWN'
        self.release_ns = None
        self.tick_latch = None
        self.safety_denials = {}
        self.last_tick = self.last_cutoff = None

    def offer(self, event):
        if not event.authenticated:
            return
        usable = event.qualified() == self.quality
        delivered = max(event.known_ns, event.arrival[1]) if usable else event.known_ns
        stable = event_key(event)
        heapq.heappush(self.pending, (delivered, stable, self.counter, event, usable))
        self.counter += 1

    def _revoke(self, stream, reason):
        self.current.pop(stream, None)
        self.velocity.pop(stream, None)
        self.velocity_reason[stream] = reason
        self.blocked[stream] = reason

    def _deny(self, reason, evidence=None):
        self.permission = evidence
        self.permission_reason = reason
        self.release_ns = None
        if self.tick_latch != 'UNSAFE':
            self.tick_latch = reason
        for stream in STREAMS:
            self._revoke(stream, 'RECOVERY_FRESHNESS')

    def _expire_permission(self, at_ns):
        if self.release_ns is not None and at_ns-self.permission.capture[0] > self.params['safety_age_ns']:
            self._deny('SAFETY_UNKNOWN', self.permission)

    def _identity_status(self, e, usable):
        stream, instance = e.stream, e.identity.instance
        owner = stream, instance
        if (*owner, e.clock_generation) in self.retired:
            return 'OUT_OF_ORDER'
        hist = self.histories.setdefault(stream, IdentityHistory())
        old = self.generations.get(owner)
        prior = hist.latest
        last = self.latest_records.get(stream)
        if (last is not None and last.identity.instance==instance and last.identity.sequence==e.identity.sequence
                and record_signature(last)!=record_signature(e)):
            return 'SOURCE_IMMUTABILITY_CONFLICT'
        status = hist.observe(e.identity)
        if status in {'OUT_OF_ORDER', 'SOURCE_UNKNOWN', 'SOURCE_IMMUTABILITY_CONFLICT'}:
            return status
        self.latest_records[stream] = e
        self.generations[owner] = e.clock_generation
        if old is not None and old != e.clock_generation:
            self.retired.add((*owner, old))
            status = 'CLOCK_GENERATION'
        previous_bound = self.latest_bounds.get(stream)
        if status != 'DUPLICATE_SOURCE':
            if (usable and previous_bound is not None and status == 'NEW' and prior is not None
                    and prior.instance == instance and e.capture[1] < previous_bound):
                status = 'CLOCK_JUMP'
            self.latest_bounds[stream] = e.capture[1] if usable else None
        return status

    def _apply_safety(self, event, usable, status, delivered):
        if not usable or status in DISCONTINUITIES or status == 'SOURCE_UNKNOWN':
            self._deny('SAFETY_UNKNOWN', event)
        elif event.safe is not True or not event.valid:
            self._deny('UNSAFE' if event.safe is False else 'SAFETY_UNKNOWN', event)
        elif event.capture[0]+self.params['safety_age_ns'] < delivered:
            self._deny('SAFETY_UNKNOWN', event)
        else:
            if self.release_ns is None:
                self.release_ns = delivered
            self.permission = event
            self.permission_reason = 'SAFE'

    def _accept_late_safety_denial(self,event,usable):
        if usable and event.valid and event.safe is True:
            return False
        instance=event.identity.instance
        owner=SAFETY,instance
        key=instance,event.clock_generation
        active=self.histories[SAFETY].latest
        return (active is not None and active.instance==instance and
                self.generations.get(owner)==event.clock_generation and
                event.identity.sequence>self.safety_denials.get(key,-1))

    def _apply(self, event, usable, delivered, updated, reasons):
        stream = event.stream
        if event.segment != 'BODY':
            reasons[stream] = 'SOURCE_UNKNOWN'
            return
        status = self._identity_status(event, usable)
        reasons[stream] = status
        late_denial=(status=='OUT_OF_ORDER' and stream==SAFETY and self._accept_late_safety_denial(event,usable))
        if status=='OUT_OF_ORDER' and not late_denial:
            return
        if status == 'DUPLICATE_SOURCE' and usable and event.valid and event.calibrated:
            if stream != SAFETY or event.safe is True:
                return
        if stream == SAFETY:
            self._apply_safety(event, usable, status, delivered)
            if self.release_ns is None:
                key=event.identity.instance,event.clock_generation
                self.safety_denials[key]=max(event.identity.sequence,self.safety_denials.get(key,-1))
            return
        if not usable or status in DISCONTINUITIES or status == 'SOURCE_UNKNOWN':
            reason = status if status in DISCONTINUITIES else ('UNSYNC' if not usable else status)
            reasons[stream] = reason
            self._revoke(stream, reason)
            return
        if not event.calibrated or not event.valid or event.pose is None:
            self._revoke(stream, invalid_reason(stream) if event.calibrated else 'CALIBRATION_INVALID')
            return
        if (self.release_ns is None or delivered <= self.release_ns or event.capture[0] < self.release_ns):
            self._revoke(stream, 'RECOVERY_FRESHNESS')
            return
        if not event.fresh:
            self._revoke(stream, 'FRESHNESS_UNKNOWN')
            return
        self._update_velocity(stream, event)
        self.current[stream] = event
        self.blocked.pop(stream, None)
        updated.add(stream)

    def _update_velocity(self, stream, event):
        prev = self.current.get(stream)
        self.velocity.pop(stream, None)
        reason = 'VELOCITY_HISTORY'
        if prev is not None:
            dt = event.capture[0]-prev.capture[1]
            bound_width = event.capture[1]-event.capture[0]+prev.capture[1]-prev.capture[0]
            if prev.version != event.version or prev.pose.shape != event.pose.shape:
                reason = 'VERSION_BOUNDARY'
            elif dt > self.params['age_ns']:
                reason = 'GAP'
            elif bound_width or dt <= 0:
                reason = 'VELOCITY_TIME_UNKNOWN'
            else:
                self.velocity[stream] = (event.pose[:,:3]-prev.pose[:,:3])/(dt*1e-9)
                reason = 'VALID'
        self.velocity_reason[stream] = reason

    def _output(self, stream, tick_ns, updated, reasons, interlock):
        item = self.current.get(stream)
        age = tick_ns-item.capture[0] if item else None
        reason = self.blocked.get(stream, reasons.get(stream,'DROPOUT'))
        if item is not None:
            reason = 'VALID' if stream in updated else 'HELD'
            if age > self.params['age_ns']:
                reason = 'STALE'
                self._revoke(stream, 'STALE')
        if interlock:
            reason = interlock
        valid = item is not None and reason in {'VALID','HELD'}
        velocity = self.velocity.get(stream) if valid else None
        velocity_reason = self.velocity_reason.get(stream,'VELOCITY_HISTORY') if valid else reason
        return dict(valid=valid,reason=reason,ref=item.ref if item else None,
                    pose=item.pose if valid else None,velocity=velocity,age_ns=age,
                    capture=item.capture if item else None,arrival=item.arrival if item else None,
                    input_reason=reasons.get(stream),version=item.version if item else None,
                    clock_generation=item.clock_generation if item else None,
                    velocity_valid=velocity is not None,velocity_age_ns=age if velocity is not None else None,
                    velocity_quality=self.quality if velocity is not None else 'UNUSABLE',
                    velocity_reason=velocity_reason)

    def step(self, tick_ns, cutoff_ns):
        if self.last_tick is not None and (tick_ns < self.last_tick or cutoff_ns < self.last_cutoff):
            raise ValueError('Past control ticks are immutable')
        while self.pending and self.pending[0][0] <= cutoff_ns:
            delivery,key,sequence,event,usable = heapq.heappop(self.pending)
            negative = (event.stream == SAFETY and (not event.valid or event.safe is not True))
            capture = event.capture[1] if usable and not negative else tick_ns
            heapq.heappush(self.capture, (capture,delivery,key,sequence,event,usable))
        visible = []
        while self.capture and self.capture[0][0] <= tick_ns:
            _,delivery,key,sequence,event,usable = heapq.heappop(self.capture)
            visible.append((delivery,key,sequence,event,usable))
        updated, reasons = set(), {}
        for delivered,_,_,e,usable in sorted(visible):
            self._expire_permission(delivered)
            self._apply(e,usable,delivered,updated,reasons)
        self._expire_permission(tick_ns)
        self.last_tick, self.last_cutoff = tick_ns, cutoff_ns
        interlock = self.tick_latch or (self.permission_reason if self.release_ns is None else None)
        if interlock:
            for stream in self.velocity:
                self.velocity_reason[stream] = 'RECOVERY_FRESHNESS'
            self.velocity.clear()
        out = {s:self._output(s,tick_ns,updated,reasons,interlock) for s in STREAMS}
        self.tick_latch = None
        joint, spans = joint_mask(out,self.params)
        return dict(tick_ns=tick_ns,cutoff_ns=cutoff_ns,streams=out,joint_valid=joint,
                    joint_reason='VALID' if joint else 'UNSYNC',skew_ns=spans,quality=self.quality,
                    safety_permission=interlock is None,safety_reason=interlock or 'SAFE',
                    safety_ref=self.permission.ref if self.permission else None,
                    safety_release_ns=self.release_ns)


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


@dataclass
class _OfflineBarrier:
    start_ns: int | None
    resume_after_ns: int
    reason: str
    end_ns: int | None = None

    def intersects(self,lo,hi):
        return (self.start_ns is None or self.start_ns<=hi) and (self.end_ns is None or lo<self.end_ns)


class OfflineAlignment:
    """Segmented interpolation in an explicit offline domain."""
    def __init__(self,events,params):
        self.params=params
        self.events={s:[] for s in (*STREAMS,SAFETY)}
        self.barriers={s:[] for s in self.events}
        self.times={}
        by_stream={s:[] for s in self.events}
        for event in events:
            if event.segment=='BODY' and event.authenticated:
                by_stream[event.stream].append(event)
        for stream,observations in by_stream.items():
            self._index_stream(stream,observations)
            self.events[stream].sort(key=lambda x:x[0].capture[1])
            self.times[stream]=[e.capture[1] for e,_,_ in self.events[stream]]

    def _index_stream(self,stream,observations):
        records={}
        conflicts=set()
        for event in sorted(observations,key=lambda e:(e.known_ns,event_key(e))):
            key=event.identity.instance,event.identity.sequence
            if key not in records:
                records[key]=event
            elif record_signature(records[key])!=record_signature(event):
                conflicts.add(key)
        hist=IdentityHistory()
        generations={}
        retired=set()
        late_denials={}
        previous=None
        barrier=None
        for (instance,seq),event in records.items():
            clock=event.clock_generation
            if (instance,clock) in retired:
                continue
            usable=event.qualified()!='UNUSABLE'
            reason=hist.observe(event.identity)
            failed=_offline_invalid(event,usable)
            if reason=='OUT_OF_ORDER':
                active=hist.latest
                if (stream!=SAFETY or not failed or active is None or active.instance!=instance
                        or generations.get(instance)!=clock or seq<=late_denials.get((instance,clock),-1)):
                    continue
            else:
                old=generations.get(instance)
                if old is not None and old!=clock:
                    retired.add((instance,old))
                    reason='CLOCK_GENERATION'
                generations[instance]=clock
            if reason=='DUPLICATE_SOURCE' and failed in {None,'FRESHNESS_UNKNOWN'} and (instance,seq) not in conflicts:
                continue
            if usable and previous is not None and event.capture[1]<previous.capture[1] and reason=='NEW':
                reason='CLOCK_JUMP'
            conflict=(instance,seq) in conflicts
            if conflict:
                failed='SAFETY_UNKNOWN' if stream==SAFETY else 'SOURCE_IMMUTABILITY_CONFLICT'
                usable=False
            elif reason in DISCONTINUITIES or reason=='SOURCE_UNKNOWN':
                failed='SAFETY_UNKNOWN' if stream==SAFETY else reason
            if failed:
                prior=previous.capture[0] if previous is not None else None
                # Arrival is only a recovery constraint. Unknown capture is never an endpoint.
                start=event.capture[0] if usable else prior
                if usable and reason in DISCONTINUITIES and prior is not None:
                    start=min(start,prior)
                resume=event.capture[1] if usable else event.known_ns
                if barrier is None:
                    barrier=_OfflineBarrier(start,resume,failed)
                    self.barriers[stream].append(barrier)
                else:
                    if start is None or (barrier.start_ns is not None and start<barrier.start_ns):
                        barrier.start_ns=start
                    barrier.resume_after_ns=max(barrier.resume_after_ns,resume)
                    if failed=='UNSAFE':
                        barrier.reason=failed
                if stream==SAFETY:
                    late_denials[instance,clock]=max(seq,late_denials.get((instance,clock),-1))
                continue
            if barrier is not None:
                if event.capture[0]<barrier.resume_after_ns:
                    continue
                barrier.end_ns=event.capture[1]
                barrier=None
            previous=event
            self.events[stream].append((event,reason,len(self.barriers[stream])))

    def _blocked(self,stream,lo,hi):
        return next((b.reason for b in self.barriers[stream] if b.intersects(lo,hi)),None)

    def _safe(self,lo,hi):
        blocked=self._blocked(SAFETY,lo,hi)
        if blocked:
            return blocked
        items=self.events[SAFETY]; times=self.times[SAFETY]
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
        blocked=self._blocked(SAFETY,tick,tick) or self._blocked(s,tick,tick)
        if blocked:
            return None,blocked
        times,items=self.times[s],self.events[s]
        k=bisect_right(times,tick)-1
        if k<0:
            return None,'DROPOUT'
        if times[k]!=tick and k+1>=len(times):
            # An invalid sample may be the missing right endpoint of this bracket.
            return None,self._blocked(s,tick,times[k]+self.params['gap_ns']) or 'DROPOUT'
        j=k if times[k]==tick else k+1
        a,ar,ae=items[k]; b,br,be=items[j]
        lo,hi=a.capture[0],b.capture[1]
        blocked=self._blocked(SAFETY,lo,hi) or self._blocked(s,lo,hi)
        if blocked:
            return None,blocked
        for reason in (ar,br):
            if reason in {'CLOCK_JUMP','SOURCE_RESTART'}:
                return None,reason
        if not a.calibrated or not b.calibrated:
            return None,'CALIBRATION_INVALID'
        if not a.valid or not b.valid or a.pose is None or b.pose is None:
            return None,invalid_reason(s)
        if not a.fresh or not b.fresh:
            return None,'FRESHNESS_UNKNOWN'
        if (ae!=be or a.identity.instance!=b.identity.instance or a.clock_generation!=b.clock_generation
                or a.quality!=b.quality or a.version!=b.version or a.pose.shape!=b.pose.shape):
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
                    time_interval=(times[k],times[j]),capture=(times[k],times[j]),version=a.version),None

    def step(self,tick_ns):
        out={}
        for s in STREAMS:
            result,why=self._bracket(s,tick_ns)
            out[s]=result or dict(valid=False,reason=why,pose=None,velocity=None,refs=None,
                                  weight=None,time_interval=None,capture=None,version=None)
        valid,spans=joint_mask(out,self.params)
        return dict(tick_ns=tick_ns,streams=out,joint_valid=valid,skew_ns=spans,quality='OFFLINE_PROXY')


def _offline_invalid(event,usable):
    if event.stream==SAFETY:
        if event.safe is False:
            return 'UNSAFE'
        return None if usable and event.safe is True and event.valid and event.fresh else 'SAFETY_UNKNOWN'
    if not usable:
        return 'UNSYNC'
    if not event.calibrated:
        return 'CALIBRATION_INVALID'
    if not event.valid or event.pose is None:
        return invalid_reason(event.stream)
    return None if event.fresh else 'FRESHNESS_UNKNOWN'
