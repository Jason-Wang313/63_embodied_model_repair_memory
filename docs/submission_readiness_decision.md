# Submission Readiness Decision

Decision: KILL_ARCHIVE.

ICLR main-conference readiness: NO.

Submission-hardening version: v4 real MuJoCo rebuild.

Reason: the paper now has real sequential MuJoCo evidence, implemented baselines, stress splits, ablations, uncertainty summaries, and paired deltas. The result is negative: embodied repair memory does not outperform nominal MPC or robust worst-case MPC and is not consistently better than simple global/last repair baselines. The mechanism fails the decisive-baseline test.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper in its current form.

Revival condition: replace kNN residual memory with a learned repair model and show statistically clear gains on public long-horizon benchmarks or hardware.
