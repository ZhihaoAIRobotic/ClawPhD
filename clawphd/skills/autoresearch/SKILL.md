---
name: autoresearch
description: Autonomous ML research loop — modify train.py, run 5-min GPU experiments, track val_bpb, iterate overnight. Invoke when the user wants to run self-directed architecture search on the autoresearch repo.
metadata: {"clawphd":{"emoji":"🔬","os":["linux"],"requires":{"bins":["uv","git"]}}}
---

# autoresearch

Autonomous ML experiment loop. You modify `train.py`, run timed GPU training, log `val_bpb`, and iterate forever — keeping improvements, reverting regressions.

The `autoresearch/` directory lives at `<workspace>/../autoresearch/` relative to the ClawPhD workspace, or wherever the user specifies. Always confirm the path before starting.

## Setup

Work with the user to:

1. **Confirm the repo path**: default is `/home/ubuntu/research/ClawPhD/autoresearch`. Verify it exists with `exec`.
2. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**: Read `README.md`, `prepare.py` (fixed — do not modify), and `train.py` (agent-editable).
5. **Verify data**: Check `~/.cache/autoresearch/` for data shards and tokenizer. If missing, tell the user to run `uv run prepare.py` first.
6. **Initialize results.tsv**: Create it with just the header row:
   ```
   commit	val_bpb	memory_gb	status	description
   ```
7. **Confirm and go**: Confirm setup, then start the loop immediately.

## Running experiments (exec + background pattern)

Since each training run takes ~5 minutes, use **background execution** to avoid blocking:

```bash
# Step 1 — Launch in background (returns immediately)
cd /home/ubuntu/research/ClawPhD/autoresearch
uv run train.py > run.log 2>&1 &
echo "Training started, PID: $!"
```

```bash
# Step 2 — Poll every ~60s until summary appears
tail -80 run.log
```

```bash
# Step 3 — Extract results once complete
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:" run.log
```

The summary block looks like:
```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
```

If `grep "^val_bpb:" run.log` returns empty after 10+ minutes, the run crashed — check `tail -50 run.log` for the stack trace.

**Timeout rule**: If `training_seconds` exceeds 360 after the summary appears, treat it as anomalous but still log it. If the process is still running after 10 minutes wall-clock, kill it: `kill <PID>`, log as `crash`, revert.

## Experiment rules

**You CAN**:
- Modify `train.py` freely: architecture, optimizer, hyperparameters, batch size, model size, anything.

**You CANNOT**:
- Modify `prepare.py` (fixed evaluation harness, data loading, constants).
- Install new packages or add dependencies.
- Change the evaluation function `evaluate_bpb`.

**Goal**: lowest `val_bpb`. Time budget is fixed at 5 minutes, so parameter count and compute efficiency both matter.

**Simplicity criterion**: All else equal, simpler code wins. A 0.001 improvement with 20 hacky lines is not worth it. Equal performance with less code is always a win.

## Logging to results.tsv

Tab-separated (NOT comma-separated). Five columns:

```
commit	val_bpb	memory_gb	status	description
```

- `commit`: 7-char git hash
- `val_bpb`: e.g. `0.997900`; use `0.000000` for crashes
- `memory_gb`: `peak_vram_mb / 1024`, rounded to 1 decimal; use `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: brief plain-text description of the change

Do **not** commit `results.tsv` (leave it untracked).

## The experiment loop

LOOP FOREVER:

1. Check git state: current branch and last commit.
2. Edit `train.py` with a new idea.
3. `git commit -m "autoresearch: <brief description>"`
4. Launch training in background (see above).
5. Poll `run.log` until the summary block appears.
6. Extract `val_bpb` and `peak_vram_mb`.
7. Log to `results.tsv`.
8. If `val_bpb` improved (lower): keep the commit, advance the branch.
9. If equal or worse: `git reset --hard HEAD~1` (revert to before this experiment).

**NEVER STOP**: Once the loop begins, do not pause to ask the user whether to continue. The user may be asleep. Run indefinitely until manually interrupted.

**When stuck**: Read papers referenced in `train.py` comments, re-read `prepare.py` for new angles, try combining previous near-misses, attempt more radical architectural changes (different depth/width ratios, different attention patterns, different optimizers).

## Reporting results via ClawPhD channels

At the end of each experiment (or every N experiments), use the `message` tool to send a summary to the user's configured channel:

```
Experiment #N: <description>
val_bpb: 0.XXXXXX → 0.YYYYYY (<+/-> delta)
Status: keep / discard
```

This way the user wakes up to a digest of overnight progress in Telegram/Slack/etc.

## Autonomous overnight use with HeartbeatService

To run the experiment loop fully autonomously (without a human kicking it off), add a task to `HEARTBEAT.md` in the workspace:

```markdown
## autoresearch
Every 10 minutes: check if an autoresearch branch is active and the training loop is running. If not, resume the loop from where it left off.
```

The `HeartbeatService` will trigger this periodically and the agent will keep the loop alive.
