"""Detect direct HMB/Teleop replay overlap by protected bag identity and exact human input payload."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from mcap.reader import make_reader

from ...hmb_reader import HMBReader, SIMULATED, checked_json
from .build import HANDOFF_SHA, REGRESSION_SHA
from .schema import SPLITS, file_sha, write_json


TOPICS = {
    "/manus_glove_left": "manus_left",
    "/manus_glove_right": "manus_right",
    "/vive/tracker/left_wrist": "tracker_left",
    "/vive/tracker/right_wrist": "tracker_right",
}


def read_payload_index(files):
    bags, contents = {}, defaultdict(set)
    for path in files:
        digest = file_sha(path)
        if digest not in REGRESSION_SHA or digest in bags:
            raise ValueError(f"Wrong or repeated protected MCAP: {path.name}")
        counts = Counter()
        with path.open("rb") as stream:
            reader = make_reader(stream)
            for sch, channel, message in reader.iter_messages(topics=tuple(TOPICS), log_time_order=False):
                if sch is None or channel.message_encoding != "cdr":
                    raise ValueError(f"Protected stream lacks declared CDR schema: {channel.topic}")
                name = TOPICS[channel.topic]
                payload = sha256(message.data).hexdigest()
                contents[(name, payload)].add(digest)
                counts[name] += 1
        if set(counts) != set(TOPICS.values()):
            raise ValueError(f"Missing protected human input stream: {path.name}")
        bags[digest] = {"basename": path.name, "core_cdr_count": dict(counts)}
    if set(bags) != REGRESSION_SHA:
        raise ValueError("Both frozen protected bags are required")
    return bags, contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--bag", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.docs / "evidence"
    handoff = checked_json(evidence / "P01_F" / "p02_handoff_manifest.json", HANDOFF_SHA)
    e1, e0 = evidence / "P01_E_E1", evidence / "P01_E_E0"
    bags, protected_payloads = read_payload_index(args.bag)
    overlap = defaultdict(Counter)
    assets_by_root = defaultdict(set)
    counts = Counter()
    total = 0
    for split in SPLITS:
        reader = HMBReader(e1 / "hmb_manifest.json", e1 / "production_plan.json", e0 / "candidate_config.json",
                           handoff["ubuntu_products"], split, SIMULATED)
        for partition in reader.keys():
            _, meta, sources, _ = reader.read(partition)
            root = meta["identity"]["root_episode_id"]
            total += 1
            for item in sources:
                record = item["record"]
                identity = record["asset_sha256"]
                stream = item["stream"]
                assets_by_root[root].add(identity)
                counts[stream] += 1
                if identity in REGRESSION_SHA:
                    overlap[root][f"asset:{identity}"] += 1
                for bag in protected_payloads.get((stream, record["cdr_sha256"]), ()):
                    overlap[root][f"exact_cdr:{bag}:{stream}"] += 1
    if total != 308 or len(assets_by_root) != 32:
        raise ValueError("Official HMB lineage scan did not cover all planned roots and partitions")
    denied_roots = sorted(overlap)
    write_json(args.out, {
        "schema": "s05.rmb.lineage_overlap.v1",
        "screen": "EXACT_FILE_SHA256_AND_EXACT_HUMAN_CDR_PAYLOAD_SHA256",
        "limits": "Byte differences do not disprove re-encoded motion or shared ancestry. Review the existing historical lineage registry separately.",
        "review_status": "READY_FOR_REVIEW",
        "roots": denied_roots,
        "asset_sha256": sorted(set(REGRESSION_SHA) | set().union(*(assets_by_root[root] for root in denied_roots))),
        "evidence": {"protected": bags, "matches": dict(overlap), "hmb_source_reference_counts": dict(counts),
                     "seen_partitions": total, "seen_roots": len(assets_by_root)},
    })
    print(f"Compared {total} signed partitions and {len(assets_by_root)} roots; {len(denied_roots)} with exact overlap", flush=True)


if __name__ == "__main__":
    main()
