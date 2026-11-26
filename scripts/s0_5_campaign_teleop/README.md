# Read-only Teleop MCAP preflight

Run on healthy storage with a Python environment containing `mcap` and its recording compression codecs.

```sh
python -B scripts/s0_5_campaign_teleop/preflight.py \
  --bag /home/ubuntu/teleop_ws/data/s0_sim_20261119_172809/s0_sim_20261119_172809_0.mcap \
  --bag /home/ubuntu/teleop_ws/data/s0_sim_20261117_181022/s0_sim_20261117_181022_0.mcap \
  --out /path/to/new-teleop-inventory.json
```

The report binds both recordings to the frozen SHA-256 values. The script inventories observed channels, embedded schemas and metadata, and MCAP log and publish stamps. Embedded payload interpretation, clock qualification, START and initial state, lineage, and the replay allowlist remain independent D1 work. Robot commands and historical feedback are audit material.

Run synthetic controls with `python -B -m unittest discover -s scripts/s0_5_campaign_teleop -v`.
