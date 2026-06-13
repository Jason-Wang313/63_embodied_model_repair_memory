# Submission Attack Log

Paper: 63 embodied_model_repair_memory

This v4 pass rebuilds the paper with real MuJoCo evidence. The result is kill/archive with real negative evidence.

## Rebuild Round 1

Attack: The previous evidence was synthetic/template-generated.

Verdict: Recovered.

Action: Replaced `src/run_experiment.py` with a real sequential MuJoCo repair-memory benchmark.

## Rebuild Round 2

Attack: No persistent deployment stream.

Verdict: Recovered.

Action: Each split/seed now uses recurring hidden deployment parameters across episodes.

## Rebuild Round 3

Attack: No implemented baselines.

Verdict: Recovered.

Action: Added nominal MPC, robust MPC, last repair, global repair, random, and oracle baselines.

## Rebuild Round 4

Attack: No ablations.

Verdict: Recovered.

Action: Added reset-memory, global, no-context, limited-capacity, nominal, and oracle ablations.

## Rebuild Round 5

Attack: Memory must beat nominal MPC.

Verdict: Failed.

Action: Kill/archive.

## Rebuild Round 6

Attack: Memory must beat robust MPC.

Verdict: Failed.

Action: Kill/archive.

## Rebuild Round 7

Attack: Memory must beat simple global/last repair.

Verdict: Failed or mixed.

Action: Kill/archive.

## Rebuild Round 8

Attack: Safety/violation behavior must improve.

Verdict: Failed.

Action: Violation rates remain high under shift.

## Terminal Decision

Decision: KILL_ARCHIVE.

The paper is no longer a synthetic archive, but the real evidence kills the claim.
