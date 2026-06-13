"""Real MuJoCo sequential repair-memory benchmark for paper 63.

The v3 script produced synthetic probability tables. This rebuild evaluates
whether persistent embodied repair memories improve later action selection after
observing model errors in repeated MuJoCo deployment streams.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


@dataclass(frozen=True)
class PhysParams:
    mass: float
    friction: float


@dataclass(frozen=True)
class Deployment:
    params: PhysParams
    distance_gain: float
    angle_bias: float
    lateral_bias: float
    obstacle_shift: tuple[float, float]


@dataclass(frozen=True)
class PushAction:
    angle: float
    offset: float
    distance: float


@dataclass(frozen=True)
class TaskSpec:
    split: str
    puck: tuple[float, float]
    target: tuple[float, float]
    obstacle: tuple[float, float]


NOMINAL = PhysParams(0.12, 0.65)
ROBUST_BRANCHES = [PhysParams(0.08, 0.25), NOMINAL, PhysParams(0.26, 1.05)]
METHODS = [
    "random_candidate",
    "nominal_mpc",
    "robust_worst_case_mpc",
    "last_repair_memory",
    "global_average_repair",
    "embodied_repair_memory",
    "oracle_hidden_deployment",
]
ABLATIONS = [
    "embodied_repair_memory",
    "reset_memory_every_episode",
    "global_average_repair",
    "no_context_features",
    "limited_memory_capacity",
    "nominal_mpc",
    "oracle_hidden_deployment",
]
SPLITS = {
    "nominal": {"mass": (0.10, 0.16), "friction": (0.50, 0.85), "gain": (0.95, 1.05), "angle": (-0.02, 0.02), "lateral": (-0.004, 0.004), "obstacle": 0.015},
    "low_friction_shift": {"mass": (0.10, 0.16), "friction": (0.12, 0.28), "gain": (0.82, 0.98), "angle": (-0.06, 0.06), "lateral": (-0.010, 0.010), "obstacle": 0.025},
    "high_friction_shift": {"mass": (0.10, 0.16), "friction": (0.95, 1.35), "gain": (0.92, 1.08), "angle": (-0.08, 0.08), "lateral": (-0.010, 0.010), "obstacle": 0.025},
    "heavy_object_shift": {"mass": (0.24, 0.40), "friction": (0.45, 0.90), "gain": (0.78, 0.95), "angle": (-0.05, 0.05), "lateral": (-0.010, 0.010), "obstacle": 0.025},
    "actuation_bias_shift": {"mass": (0.10, 0.18), "friction": (0.45, 0.85), "gain": (0.70, 1.20), "angle": (-0.16, 0.16), "lateral": (-0.020, 0.020), "obstacle": 0.035},
    "combined_shift": {"mass": (0.06, 0.42), "friction": (0.12, 1.35), "gain": (0.68, 1.22), "angle": (-0.18, 0.18), "lateral": (-0.025, 0.025), "obstacle": 0.055},
}
MODEL_CACHE: dict[PhysParams, mujoco.MjModel] = {}
PUCK_RADIUS = 0.045
OBSTACLE_RADIUS = 0.055
SUCCESS_RADIUS = 0.075


def make_model(params: PhysParams) -> mujoco.MjModel:
    cached = MODEL_CACHE.get(params)
    if cached is not None:
        return cached
    xml = f"""
    <mujoco model="repair_memory_push">
      <option timestep="0.006" gravity="0 0 -9.81" integrator="RK4"/>
      <default>
        <geom condim="3" solref="0.006 1" solimp="0.9 0.95 0.001" friction="{params.friction} 0.004 0.0001"/>
      </default>
      <worldbody>
        <light pos="0 0 1"/>
        <geom name="floor" type="plane" size="1.2 1.2 0.02" rgba="0.75 0.75 0.75 1" friction="{params.friction} 0.004 0.0001"/>
        <body name="puck" pos="0 0 0.026">
          <freejoint name="puck_free"/>
          <geom name="puck_geom" type="cylinder" size="{PUCK_RADIUS} 0.025" mass="{params.mass}" rgba="0.1 0.3 0.9 1" friction="{params.friction} 0.004 0.0001"/>
        </body>
        <body name="pusher" pos="0 0 0.042">
          <joint name="px" type="slide" axis="1 0 0" damping="8"/>
          <joint name="py" type="slide" axis="0 1 0" damping="8"/>
          <geom name="pusher_geom" type="sphere" size="0.026" mass="0.25" rgba="0.9 0.25 0.1 1" friction="1.2 0.004 0.0001"/>
        </body>
        <body name="obstacle" mocap="true" pos="0.20 0 0.040">
          <geom name="obstacle_geom" type="cylinder" size="{OBSTACLE_RADIUS} 0.040" rgba="0.05 0.05 0.05 1" friction="1.2 0.004 0.0001"/>
        </body>
      </worldbody>
      <actuator>
        <position name="px_ctrl" joint="px" kp="520" ctrlrange="-1 1"/>
        <position name="py_ctrl" joint="py" kp="520" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    MODEL_CACHE[params] = model
    return model


def set_state(data: mujoco.MjData, puck_xy: np.ndarray, pusher_xy: np.ndarray, obstacle_xy: np.ndarray) -> None:
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[0:7] = [float(puck_xy[0]), float(puck_xy[1]), 0.026, 1, 0, 0, 0]
    data.qpos[7:9] = [float(pusher_xy[0]), float(pusher_xy[1])]
    data.ctrl[0:2] = pusher_xy
    data.mocap_pos[0] = [float(obstacle_xy[0]), float(obstacle_xy[1]), 0.040]
    data.mocap_quat[0] = [1, 0, 0, 0]


def action_path(puck_xy: np.ndarray, action: PushAction, deployment: Deployment | None = None) -> tuple[np.ndarray, np.ndarray]:
    gain = deployment.distance_gain if deployment else 1.0
    angle_bias = deployment.angle_bias if deployment else 0.0
    lateral_bias = deployment.lateral_bias if deployment else 0.0
    angle = action.angle + angle_bias
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    offset = action.offset + lateral_bias
    start = puck_xy - 0.125 * direction + offset * normal
    end = puck_xy + (action.distance * gain) * direction + offset * normal
    return start, end


def shifted_obstacle(base: np.ndarray, deployment: Deployment | None) -> np.ndarray:
    if deployment is None:
        return base
    return base + np.array(deployment.obstacle_shift, dtype=float)


def rollout_push(
    params: PhysParams,
    puck_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    action: PushAction,
    deployment: Deployment | None = None,
) -> dict:
    model = make_model(params)
    data = mujoco.MjData(model)
    obstacle = shifted_obstacle(obstacle_xy, deployment)
    start, end = action_path(puck_xy, action, deployment)
    set_state(data, puck_xy, start, obstacle)
    mujoco.mj_forward(model, data)
    min_obstacle_dist = float(np.linalg.norm(puck_xy - obstacle))
    effort = 0.0
    last = start
    for i in range(50):
        alpha = (i + 1) / 50.0
        target = (1 - alpha) * start + alpha * end
        effort += float(np.linalg.norm(target - last))
        last = target
        data.ctrl[0] = float(target[0])
        data.ctrl[1] = float(target[1])
        mujoco.mj_step(model, data)
        min_obstacle_dist = min(min_obstacle_dist, float(np.linalg.norm(np.array(data.qpos[0:2]) - obstacle)))
    for _ in range(16):
        data.ctrl[0] = float(end[0])
        data.ctrl[1] = float(end[1])
        mujoco.mj_step(model, data)
        min_obstacle_dist = min(min_obstacle_dist, float(np.linalg.norm(np.array(data.qpos[0:2]) - obstacle)))
    final_xy = np.array(data.qpos[0:2], dtype=float)
    violation = float(min_obstacle_dist < (PUCK_RADIUS + OBSTACLE_RADIUS + 0.006))
    return {"final_xy": final_xy, "violation": violation, "effort": effort, "min_obstacle_dist": min_obstacle_dist}


def sample_deployment(split: str, seed: int) -> Deployment:
    rng = random.Random(6300001 + 1597 * seed + sum(ord(c) for c in split))
    cfg = SPLITS[split]
    obstacle_mag = cfg["obstacle"]
    return Deployment(
        params=PhysParams(rng.uniform(*cfg["mass"]), rng.uniform(*cfg["friction"])),
        distance_gain=rng.uniform(*cfg["gain"]),
        angle_bias=rng.uniform(*cfg["angle"]),
        lateral_bias=rng.uniform(*cfg["lateral"]),
        obstacle_shift=(rng.uniform(-obstacle_mag, obstacle_mag), rng.uniform(-obstacle_mag, obstacle_mag)),
    )


def sample_task(split: str, seed: int, episode: int) -> TaskSpec:
    rng = random.Random(6309103 + 100003 * seed + 7919 * episode + sum(ord(c) for c in split))
    puck = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)], dtype=float)
    target_angle = rng.uniform(-0.70, 0.70)
    target_radius = rng.uniform(0.27, 0.43)
    target = puck + target_radius * np.array([math.cos(target_angle), math.sin(target_angle)], dtype=float)
    midpoint = 0.50 * (puck + target)
    normal = np.array([-math.sin(target_angle), math.cos(target_angle)], dtype=float)
    obstacle = midpoint + rng.choice([-1, 1]) * rng.uniform(0.05, 0.13) * normal
    return TaskSpec(split, tuple(puck), tuple(target), tuple(obstacle))


def candidate_actions(puck_xy: np.ndarray, target_xy: np.ndarray) -> list[PushAction]:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    remaining = float(np.linalg.norm(target_xy - puck_xy))
    actions: list[PushAction] = []
    for deg in [-45, -25, -10, 0, 10, 25, 45]:
        for scale in [0.78, 1.08]:
            actions.append(PushAction(base + math.radians(deg), 0.0, max(0.16, min(0.54, scale * remaining))))
    return actions


def line_clearance(start: np.ndarray, end: np.ndarray, obstacle: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment)) + 1e-8
    t = max(0.0, min(1.0, float(np.dot(obstacle - start, segment) / denom)))
    closest = start + t * segment
    return float(np.linalg.norm(closest - obstacle))


def action_features(puck_xy: np.ndarray, target_xy: np.ndarray, obstacle_xy: np.ndarray, action: PushAction, nominal_final: np.ndarray) -> np.ndarray:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    angle_rel = math.atan2(math.sin(action.angle - base), math.cos(action.angle - base))
    direction = np.array([math.cos(action.angle), math.sin(action.angle)], dtype=float)
    geom_end = puck_xy + action.distance * direction
    init_dist = float(np.linalg.norm(target_xy - puck_xy))
    return np.array(
        [
            math.sin(angle_rel),
            math.cos(angle_rel),
            action.distance,
            init_dist,
            float(np.linalg.norm(target_xy - geom_end)),
            float(np.linalg.norm(target_xy - nominal_final)),
            line_clearance(puck_xy, geom_end, obstacle_xy),
            float(np.linalg.norm(obstacle_xy - puck_xy)),
            float(np.linalg.norm(obstacle_xy - target_xy)),
        ],
        dtype=np.float32,
    )


def energy(final_xy: np.ndarray, target_xy: np.ndarray, violation: float, effort: float) -> float:
    return float(np.linalg.norm(final_xy - target_xy)) + 0.30 * float(violation) + 0.03 * effort


class RepairMemory:
    def __init__(self, mode: str, capacity: int = 256, use_context: bool = True) -> None:
        self.mode = mode
        self.capacity = capacity
        self.use_context = use_context
        self.features: list[np.ndarray] = []
        self.residuals: list[np.ndarray] = []
        self.violation_residuals: list[float] = []

    def reset(self) -> None:
        self.features.clear()
        self.residuals.clear()
        self.violation_residuals.clear()

    def add(self, feature: np.ndarray, residual: np.ndarray, violation_residual: float) -> None:
        feat = feature if self.use_context else np.zeros_like(feature)
        self.features.append(feat.astype(np.float32))
        self.residuals.append(residual.astype(np.float32))
        self.violation_residuals.append(float(violation_residual))
        if len(self.features) > self.capacity:
            self.features = self.features[-self.capacity :]
            self.residuals = self.residuals[-self.capacity :]
            self.violation_residuals = self.violation_residuals[-self.capacity :]

    def predict(self, feature: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.features:
            return np.zeros(2, dtype=float), 0.0
        if self.mode == "last":
            return np.array(self.residuals[-1], dtype=float), self.violation_residuals[-1]
        if self.mode == "global":
            return np.mean(np.stack(self.residuals), axis=0), float(np.mean(self.violation_residuals))
        if len(self.features) < 4:
            return np.zeros(2, dtype=float), 0.0
        feat = feature if self.use_context else np.zeros_like(feature)
        feats = np.stack(self.features)
        scale = feats.std(axis=0) + 0.05
        d = np.linalg.norm((feats - feat) / scale, axis=1)
        k = min(5, len(d))
        idx = np.argsort(d)[:k]
        weights = np.exp(-d[idx])
        weights /= max(weights.sum(), 1e-8)
        residual = np.sum(np.stack([self.residuals[i] for i in idx]) * weights[:, None], axis=0)
        violation = float(np.sum(np.array([self.violation_residuals[i] for i in idx]) * weights))
        trust = min(0.85, len(self.features) / 18.0) * math.exp(-0.12 * float(d[idx[0]]))
        return trust * residual, trust * violation


def prepare_candidates(task: TaskSpec, deployment: Deployment) -> list[dict]:
    puck = np.array(task.puck, dtype=float)
    target = np.array(task.target, dtype=float)
    obstacle = np.array(task.obstacle, dtype=float)
    actions = candidate_actions(puck, target)
    rows = []
    for idx, action in enumerate(actions):
        nominal = rollout_push(NOMINAL, puck, obstacle, action, None)
        true = rollout_push(deployment.params, puck, obstacle, action, deployment)
        robust_scores = []
        for branch in ROBUST_BRANCHES:
            pred = rollout_push(branch, puck, obstacle, action, None)
            robust_scores.append(energy(pred["final_xy"], target, pred["violation"], pred["effort"]))
        feat = action_features(puck, target, obstacle, action, nominal["final_xy"])
        rows.append(
            {
                "action_idx": idx,
                "action": action,
                "feature": feat,
                "target_xy": target,
                "nominal": nominal,
                "true": true,
                "robust_score": max(robust_scores),
                "nominal_energy": energy(nominal["final_xy"], target, nominal["violation"], nominal["effort"]),
                "true_energy": energy(true["final_xy"], target, true["violation"], true["effort"]),
            }
        )
    return rows


def choose_candidate(method: str, rows: list[dict], memory: RepairMemory | None, rng: random.Random) -> int:
    if method == "random_candidate":
        chosen = rng.randrange(len(rows))
    elif method == "nominal_mpc" or method == "reset_memory_every_episode":
        chosen = int(np.argmin([row["nominal_energy"] for row in rows]))
    elif method == "robust_worst_case_mpc":
        chosen = int(np.argmin([row["robust_score"] for row in rows]))
    elif method == "oracle_hidden_deployment":
        chosen = int(np.argmin([row["true_energy"] for row in rows]))
    else:
        scores = []
        for row in rows:
            residual, violation_residual = memory.predict(row["feature"]) if memory is not None else (np.zeros(2), 0.0)
            predicted_final = row["nominal"]["final_xy"] + residual
            predicted_violation = max(0.0, min(1.0, float(row["nominal"]["violation"]) + violation_residual))
            scores.append(energy(predicted_final, row["target_xy"], predicted_violation, row["nominal"]["effort"]))
        chosen = int(np.argmin(scores))
    return chosen


def make_memory(method: str) -> RepairMemory | None:
    if method in {"last_repair_memory"}:
        return RepairMemory("last", capacity=1)
    if method in {"global_average_repair"}:
        return RepairMemory("global", capacity=512)
    if method in {"embodied_repair_memory"}:
        return RepairMemory("knn", capacity=512, use_context=True)
    if method in {"no_context_features"}:
        return RepairMemory("knn", capacity=512, use_context=False)
    if method in {"limited_memory_capacity"}:
        return RepairMemory("knn", capacity=12, use_context=True)
    if method in {"reset_memory_every_episode"}:
        return RepairMemory("knn", capacity=512, use_context=True)
    return None


def run_stream(split: str, seed: int, episodes: int, method: str, ablation: bool = False) -> list[dict]:
    rng_seed = 6389 + 99991 * seed + sum(ord(c) for c in split) + sum(ord(c) for c in method)
    random.seed(rng_seed)
    deployment = sample_deployment(split, seed)
    memory = make_memory(method)
    rows = []
    for episode in range(episodes):
        if method == "reset_memory_every_episode" and memory is not None:
            memory.reset()
        task = sample_task(split, seed, episode)
        candidates = prepare_candidates(task, deployment)
        chosen = choose_candidate(method, candidates, memory, random.Random(rng_seed + episode))
        selected = candidates[chosen]
        puck = np.array(task.puck, dtype=float)
        target = np.array(task.target, dtype=float)
        true_out = selected["true"]
        nominal_out = selected["nominal"]
        final_distance = float(np.linalg.norm(true_out["final_xy"] - target))
        oracle_energy = min(float(row["true_energy"]) for row in candidates)
        if memory is not None:
            residual = true_out["final_xy"] - nominal_out["final_xy"]
            violation_residual = float(true_out["violation"] - nominal_out["violation"])
            memory.add(selected["feature"], residual, violation_residual)
        rows.append(
            {
                "seed": seed,
                "episode": episode,
                "split": split,
                "method": method,
                "true_mass": deployment.params.mass,
                "true_friction": deployment.params.friction,
                "distance_gain": deployment.distance_gain,
                "angle_bias": deployment.angle_bias,
                "initial_distance": float(np.linalg.norm(target - puck)),
                "chosen_action": chosen,
                "success": float(final_distance <= SUCCESS_RADIUS and true_out["violation"] < 0.5),
                "final_distance": final_distance,
                "violation": float(true_out["violation"]),
                "energy": float(selected["true_energy"]),
                "oracle_energy": oracle_energy,
                "energy_regret": float(selected["true_energy"] - oracle_energy),
                "ablation": ablation,
            }
        )
    return rows


def run_method_set(split: str, seed: int, episodes: int, methods: list[str], ablation: bool = False) -> list[dict]:
    deployment = sample_deployment(split, seed)
    memories = {method: make_memory(method) for method in methods}
    rngs = {method: random.Random(6389 + 99991 * seed + sum(ord(c) for c in split) + sum(ord(c) for c in method)) for method in methods}
    rows = []
    for episode in range(episodes):
        task = sample_task(split, seed, episode)
        candidates = prepare_candidates(task, deployment)
        target = np.array(task.target, dtype=float)
        puck = np.array(task.puck, dtype=float)
        oracle_energy = min(float(row["true_energy"]) for row in candidates)
        for method in methods:
            memory = memories[method]
            if method == "reset_memory_every_episode" and memory is not None:
                memory.reset()
            chosen = choose_candidate(method, candidates, memory, rngs[method])
            selected = candidates[chosen]
            true_out = selected["true"]
            nominal_out = selected["nominal"]
            final_distance = float(np.linalg.norm(true_out["final_xy"] - target))
            if memory is not None:
                residual = true_out["final_xy"] - nominal_out["final_xy"]
                violation_residual = float(true_out["violation"] - nominal_out["violation"])
                memory.add(selected["feature"], residual, violation_residual)
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "split": split,
                    "method": method,
                    "true_mass": deployment.params.mass,
                    "true_friction": deployment.params.friction,
                    "distance_gain": deployment.distance_gain,
                    "angle_bias": deployment.angle_bias,
                    "initial_distance": float(np.linalg.norm(target - puck)),
                    "chosen_action": chosen,
                    "success": float(final_distance <= SUCCESS_RADIUS and true_out["violation"] < 0.5),
                    "final_distance": final_distance,
                    "violation": float(true_out["violation"]),
                    "energy": float(selected["true_energy"]),
                    "oracle_energy": oracle_energy,
                    "energy_regret": float(selected["true_energy"] - oracle_energy),
                    "ablation": ablation,
                }
            )
    return rows


def ci95(vals: Iterable[float]) -> float:
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / math.sqrt(len(vals))


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        successes = [float(v["success"]) for v in vals]
        distances = [float(v["final_distance"]) for v in vals]
        violations = [float(v["violation"]) for v in vals]
        regrets = [float(v["energy_regret"]) for v in vals]
        summary = {k: key[i] for i, k in enumerate(keys)}
        summary.update(
            {
                "episodes": len(vals),
                "success_rate": mean(successes),
                "success_ci95": ci95(successes),
                "final_distance_mean": mean(distances),
                "final_distance_ci95": ci95(distances),
                "violation_rate": mean(violations),
                "violation_ci95": ci95(violations),
                "energy_regret_mean": mean(regrets),
                "energy_regret_ci95": ci95(regrets),
            }
        )
        out.append(summary)
    return out


def learning_curve(rows: list[dict]) -> list[dict]:
    out = []
    max_episode = max(int(row["episode"]) for row in rows)
    bins = [(0, max_episode // 3), (max_episode // 3 + 1, 2 * max_episode // 3), (2 * max_episode // 3 + 1, max_episode)]
    labels = ["early", "middle", "late"]
    for label, (lo, hi) in zip(labels, bins):
        subset = [row for row in rows if lo <= int(row["episode"]) <= hi]
        out.extend(summarize([{**row, "phase": label} for row in subset], ["phase", "split", "method"]))
    return out


def paired_stats(rows: list[dict]) -> list[dict]:
    proposed = "embodied_repair_memory"
    baselines = ["nominal_mpc", "robust_worst_case_mpc", "last_repair_memory", "global_average_repair", "oracle_hidden_deployment"]
    by_key: dict[tuple, dict] = {}
    for row in rows:
        by_key.setdefault((row["split"], row["seed"], row["episode"]), {})[row["method"]] = row
    out = []
    for split in sorted({row["split"] for row in rows}):
        cases = [methods for key, methods in by_key.items() if key[0] == split and proposed in methods]
        for baseline in baselines:
            paired = [(methods[proposed], methods[baseline]) for methods in cases if baseline in methods]
            if not paired:
                continue
            success_delta = [float(p["success"]) - float(b["success"]) for p, b in paired]
            regret_delta = [float(b["energy_regret"]) - float(p["energy_regret"]) for p, b in paired]
            violation_delta = [float(p["violation"]) - float(b["violation"]) for p, b in paired]
            out.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_episodes": len(paired),
                    "success_delta_mean": f"{mean(success_delta):.4f}",
                    "success_delta_ci95": f"{ci95(success_delta):.4f}",
                    "regret_improvement_mean": f"{mean(regret_delta):.4f}",
                    "regret_improvement_ci95": f"{ci95(regret_delta):.4f}",
                    "violation_delta_mean": f"{mean(violation_delta):.4f}",
                    "violation_delta_ci95": f"{ci95(violation_delta):.4f}",
                }
            )
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        clean = dict(row)
        for key, value in row.items():
            if isinstance(value, float):
                clean[key] = f"{value:.4f}"
        formatted.append(clean)
    return formatted


def plot_results(metrics: list[dict], learning: list[dict], ablation: list[dict]) -> None:
    splits = sorted({row["split"] for row in metrics})
    methods = ["nominal_mpc", "robust_worst_case_mpc", "last_repair_memory", "global_average_repair", "embodied_repair_memory", "oracle_hidden_deployment"]
    labels = ["Nominal", "Robust", "Last", "Global", "RepairMem", "Oracle"]
    x = np.arange(len(splits))
    width = 0.13
    plt.figure(figsize=(12, 4.8))
    for idx, method in enumerate(methods):
        vals = [float(next(row["success_rate"] for row in metrics if row["split"] == split and row["method"] == method)) for split in splits]
        plt.bar(x + (idx - 2.5) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=20, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.02)
    plt.title("Repair memory success by deployment split")
    plt.legend(ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_success_by_split.png", dpi=180)
    plt.close()

    phases = ["early", "middle", "late"]
    plt.figure(figsize=(8, 4.8))
    for method, label in [("nominal_mpc", "Nominal"), ("robust_worst_case_mpc", "Robust"), ("global_average_repair", "Global"), ("embodied_repair_memory", "RepairMem")]:
        vals = []
        for phase in phases:
            phase_vals = [float(row["success_rate"]) for row in learning if row["phase"] == phase and row["method"] == method]
            vals.append(mean(phase_vals) if phase_vals else 0.0)
        plt.plot(phases, vals, marker="o", label=label)
    plt.ylabel("Average success across splits")
    plt.title("Learning curve over repeated deployment tasks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_learning_curve.png", dpi=180)
    plt.close()

    order = sorted(ablation, key=lambda row: float(row["energy_regret_mean"]))
    plt.figure(figsize=(9, 4.8))
    plt.barh([row["method"] for row in order], [float(row["energy_regret_mean"]) for row in order])
    plt.xlabel("Energy regret vs oracle")
    plt.title("Combined-shift repair-memory ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_ablation_regret.png", dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    raw_rows: list[dict] = []
    for split in args.splits:
        for seed in range(args.seeds):
            raw_rows.extend(run_method_set(split, seed, args.episodes, METHODS, ablation=False))
        write_rows(RESULTS / "repair_memory_raw.partial.csv", format_rows(raw_rows))
        write_rows(RESULTS / "repair_memory_metrics.partial.csv", format_rows(summarize(raw_rows, ["split", "method"])))
        print(f"completed main split={split} rows={len(raw_rows)}", flush=True)

    ablation_rows: list[dict] = []
    for seed in range(args.seeds):
        ablation_rows.extend(run_method_set("combined_shift", seed, args.episodes, ABLATIONS, ablation=True))
        write_rows(RESULTS / "repair_memory_ablation.partial.csv", format_rows(summarize(ablation_rows, ["method"])))
        print(f"completed ablation seed={seed} rows={len(ablation_rows)}", flush=True)

    main_summary = summarize(raw_rows, ["split", "method"])
    seed_summary = summarize(raw_rows, ["split", "method", "seed"])
    ablation_summary = summarize(ablation_rows, ["method"])
    learning = learning_curve(raw_rows)
    pairwise = paired_stats(raw_rows)

    write_rows(RESULTS / "repair_memory_raw.csv", format_rows(raw_rows))
    write_rows(RESULTS / "repair_memory_metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "repair_memory_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "repair_memory_ablation.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "repair_memory_learning_curve.csv", format_rows(learning))
    write_rows(RESULTS / "repair_memory_pairwise.csv", pairwise)
    write_rows(RESULTS / "metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "raw_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "ablation_metrics.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "stress_sweep.csv", format_rows(learning))
    write_rows(RESULTS / "pairwise_stats.csv", pairwise)
    write_rows(FIGURES / "stress_curve_data.csv", format_rows(learning))
    negative_cases = [
        {"case": "nonrecurring_random_damage", "observed": "repair memory can retrieve irrelevant residuals", "paper_status": "limitation"},
        {"case": "semantic_goal_error", "observed": "physical repair memory cannot fix wrong task specification", "paper_status": "limitation"},
        {"case": "custom_mujoco_only", "observed": "evidence supports strong-revise at best without public benchmark/hardware", "paper_status": "limitation"},
    ]
    write_rows(RESULTS / "negative_cases.csv", negative_cases)
    plot_results(main_summary, learning, ablation_summary)
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Real MuJoCo sequential repair-memory benchmark for paper 63\n")
        f.write(f"seeds={args.seeds} episodes_per_seed_split_method={args.episodes} splits={','.join(args.splits)}\n")
        for row in main_summary:
            if row["method"] in {"embodied_repair_memory", "global_average_repair", "robust_worst_case_mpc", "oracle_hidden_deployment"}:
                f.write(
                    f"{row['split']} {row['method']} success={row['success_rate']:.3f}+/-{row['success_ci95']:.3f} "
                    f"regret={row['energy_regret_mean']:.3f}+/-{row['energy_regret_ci95']:.3f} violation={row['violation_rate']:.3f}\n"
                )
    print(f"wrote real repair-memory benchmark results to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
