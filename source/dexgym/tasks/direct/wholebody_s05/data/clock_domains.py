"""Event-time bounds with explicit quality and causal synchronization anchors."""
from dataclasses import dataclass
from math import ceil,floor


DOMAIN_FIELDS={
    'source_capture':('physical_device_capture',None,'No timestamp unless a device observation is bound'),
    'sdk_source':('manus_compressed',None,'SDK publishTime packed calendar or timecode'),
    'sdk_receive':('publisher_steady_clock','us','Monotonic local arrival in publisher'),
    'ros_header':('message_ros_clock','ns','Publisher ROS timestamp or copied tracker input header'),
    'mcap_log':('collector_host_realtime','ns','Recorder envelope log_time'),
    'mcap_publish':('collector_host_realtime','ns','Native writer stores the same envelope timestamp payload'),
    'collector_receive':('collector_ros_clock','ns','SampleObservation receive_ros_timestamp_ns'),
    'collector_monotonic':('collector_host_monotonic','ns','SampleObservation host_observed_monotonic_ns'),
    'collector_realtime':('collector_host_realtime','ns','SampleObservation host_realtime_ns'),
    'controller_receive':('controller_host_monotonic','ns','Actual controller input arrival if instrumented'),
    'control_tick':('controller_host_monotonic','ns','Actual scheduler query if instrumented'),
    'command_issue':('controller_host_monotonic','ns','Actual command release if instrumented'),
    'robot_feedback':('device_measurement',None,'Actual calibrated measurement if instrumented'),
}


@dataclass(frozen=True)
class Anchor:
    source_ns: int
    lower_ns: int
    upper_ns: int
    available_ns: int
    evidence: str


class PrefixClock:
    """Build intersected causal bounds given independently witnessed slope limits."""
    def __init__(self,slope_min=1.0,slope_max=1.0,evidence=None):
        if not (0<slope_min<=slope_max):
            raise ValueError('Incorrect clock slope bounds')
        self.slope=(slope_min,slope_max)
        self.evidence=evidence
        self.anchors=[]

    def add(self,anchor):
        if anchor.lower_ns>anchor.upper_ns or not anchor.evidence:
            raise ValueError('Unbounded observation')
        self.anchors.append(anchor)

    def bounds(self,source_ns,cutoff_ns):
        eligible=[a for a in self.anchors if a.available_ns<=cutoff_ns]
        interval=[]
        for a in eligible:
            x=source_ns-a.source_ns
            y,z=(x*v for v in self.slope)
            interval.append((a.lower_ns+floor(min(y,z)),a.upper_ns+ceil(max(y,z))))
        if not self.evidence or not interval:
            return None,'UNUSABLE',len(eligible)
        lo=max(x for x,_ in interval); hi=min(y for _,y in interval)
        return ((lo,hi),'VERIFIED',len(eligible)) if lo<=hi else (None,'CLOCK_JUMP',len(eligible))


def qualify(capture,arrival,requested,independent_capture_proof=None,independent_controller_proof=None):
    valid = lambda x: (x is not None and len(x)==2 and all(type(v) is int for v in x) and x[0]<=x[1])
    if not valid(capture) or not valid(arrival):
        return 'UNUSABLE'
    if requested=='VERIFIED':
        return 'VERIFIED' if independent_capture_proof and independent_controller_proof else 'UNUSABLE'
    return requested if requested in {'PROXY','SIMULATED'} else 'UNUSABLE'
