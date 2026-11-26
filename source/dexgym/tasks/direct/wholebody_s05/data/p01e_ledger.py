"""Append-only, fsynced per-partition state machine for P01-E."""
from datetime import datetime, timezone
import json
import os
from hashlib import sha256

from .hmb_build import json_bytes


TRANSITIONS={None:{'PLANNED'},'PLANNED':{'RUNNING','PAUSED'},
             'RUNNING':{'PUBLISHED','QUARANTINED','FAILED','PAUSED','RESUMED'},
             'FAILED':{'PAUSED','RESUMED'},'PAUSED':{'RESUMED'},'RESUMED':{'RUNNING','PAUSED'},
             'PUBLISHED':set(),'QUARANTINED':set()}


class Ledger:
    def __init__(self,path,scope,plan):
        self.path=path
        self.scope=scope
        self.plan=plan
        self.latest={}
        self.publications={}
        self.entries=0
        self.plan_by_key={row['key']:row for row in plan}
        if len(self.plan_by_key)!=len(plan):
            raise ValueError('Duplicate planned partition')
        self.plan_digest=sha256(json_bytes(plan)).hexdigest()
        if path.exists():
            for line in path.read_bytes().splitlines():
                self._observe(json.loads(line))
        for row in plan:
            if row['key'] not in self.latest:
                self.add(row['key'],'PLANNED')

    def _observe(self,event):
        k=event['key']
        before=self.latest[k]['status'] if k in self.latest else None
        if (k not in self.plan_by_key or event['plan_sha256']!=self.plan_digest or
                event['scope']!=self.scope or event['ordinal']!=self.entries or
                event['partition']!=self.plan_by_key[k] or event['status'] not in TRANSITIONS[before]):
            raise ValueError('Ledger provenance or transition mismatch')
        self.latest[k]=event
        self.entries+=1
        if event['status']=='PUBLISHED':
            self.publications[k]=event['evidence']

    def add(self,key,status,**evidence):
        event=dict(key=key,status=status,scope=self.scope,partition=self.plan_by_key[key],
                   plan_sha256=self.plan_digest,ordinal=self.entries,
                   at=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),evidence=evidence)
        self._observe(event)
        with self.path.open('ab') as stream:
            stream.write(json_bytes(event))
            stream.flush()
            os.fsync(stream.fileno())

    def completed_index(self):
        states=[]
        for planned in self.plan:
            actual=self.latest[planned['key']]
            if actual['status'] not in {'PUBLISHED','QUARANTINED'}:
                raise ValueError('At least one planned partition lacks terminal disposition')
            states.append(dict(partition=planned,status=actual['status'],evidence=actual['evidence']))
        return states
