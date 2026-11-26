"""Separate source input identity from ROS publication and reader order."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    instance: str
    sequence: int
    source_stamp: int
    receive_stamp: int
    fingerprint: str
    component_ms: int | None = None


class IdentityHistory:
    def __init__(self):
        self.latest=None
        self.previous_instances=set()

    def observe(self,new):
        old=self.latest
        if not new.instance or new.sequence<0 or new.source_stamp<0 or new.receive_stamp<0:
            return 'SOURCE_UNKNOWN'
        if old is None:
            self.latest=new
            return 'NEW'
        if new.instance!=old.instance:
            if new.instance in self.previous_instances:
                return 'OUT_OF_ORDER'
            self.previous_instances.add(old.instance)
            self.latest=new
            return 'SOURCE_RESTART'
        if new.sequence<=old.sequence:
            return 'OUT_OF_ORDER'
        if (new.source_stamp,new.receive_stamp,new.fingerprint)==(old.source_stamp,old.receive_stamp,old.fingerprint):
            # Writer sequence is the latest observed transport event, capture identity is unchanged.
            self.latest=new
            return 'DUPLICATE_SOURCE'
        if new.receive_stamp<old.receive_stamp or (
                new.component_ms is not None and old.component_ms is not None and new.component_ms<old.component_ms):
            self.latest=new
            return 'CLOCK_JUMP'
        if new.source_stamp==old.source_stamp and new.receive_stamp==old.receive_stamp:
            return 'SOURCE_IMMUTABILITY_CONFLICT'
        self.latest=new
        return 'NEW'
