# Child Status 63

Current stage: expanded-standard v5 frozen MuJoCo rebuild complete
Last update: 2026-06-20 01:01:40 +08:00
PDF: C:/Users/wangz/Downloads/63.pdf
PDF SHA256: E4FD93FAE7E60EF0260DDC75512183E19C80FBB1EFD097DCD542EE972DC02978
GitHub: https://github.com/Jason-Wang313/63_embodied_model_repair_memory
Submission-hardening version: v5 expanded
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Reason: CORM v5 was rebuilt under a frozen hostile-review protocol with 25,344 main MuJoCo rows, 5,632 ablation rows, 9 stress splits, strong baselines, paired bootstrap/sign-flip statistics, ablations, generated tables, bright boxed citation links, and a 25-page Downloads-only PDF. The result remains KILL_ARCHIVE because CORM fails the predefined strong-baseline gates: aggregate success is below old kNN repair memory and online ridge residual, regret does not improve over robust MPC or online ridge, and nonstationary stress increases violations relative to robust MPC.
