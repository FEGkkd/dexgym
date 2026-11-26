"""Exercise the DCR-002 state gate without changing persisted approvals."""
from copy import deepcopy
from datetime import datetime,timedelta
import json
from pathlib import Path
import runpy


DOCS=Path('docs/s0-5')


def main():
    checker=runpy.run_path(str(DOCS/'evidence/P00/check_documents.py'))
    state=json.loads((DOCS/'phase_state.json').read_bytes())
    checker['phase_states'](state)
    wrong_digest=deepcopy(state)
    wrong_digest['p01_subphase_events'][-1]['evidence_sha256']='0'*64
    forged_clock=deepcopy(state)
    forged_clock['clock_qualifications']['C2-RT']='ACCEPTED'
    next_phase=deepcopy(state)
    last=next_phase['p01_subphase_events'][-1]
    at=(datetime.fromisoformat(last['at'])+timedelta(seconds=1)).isoformat()
    next_phase['p01_subphase_events'].append(dict(
        phase='P01-D',**{'from':'NOT_STARTED'},to='IN_PROGRESS',at=at,
        authorization_ref='negative-control-only',
        spec_sha256=next_phase['dependency_dcr']['active_spec_sha256']))
    next_phase['p01_subphases']['P01-D']='IN_PROGRESS'
    for edited,expected in ((wrong_digest,'technical receipt hash mismatch'),
                            (forged_clock,'qualification state differs from history'),
                            (next_phase,'P01-D requires P01-C ACCEPTED')):
        try:
            checker['phase_states'](edited)
        except ValueError as exc:
            if expected not in str(exc):
                raise
        else:
            raise ValueError(f'An invalid stage change was accepted: {expected}')
        print(f'PASS negative control: {expected}')


if __name__=='__main__':
    main()
