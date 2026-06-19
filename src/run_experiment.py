"""MuJoCo repeated-deployment repair-memory benchmark for Paper 63.

Version 5 replaces the falsified kNN-only repair memory with a CPU-light
calibrated online residual model.  The runner is intentionally hostile to the
method: all policies share the same candidate action set, strong baselines are
run on the same tasks, and the final protocol reports safety and nonstationary
failures instead of hiding them.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


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


@dataclass(frozen=True)
class MemoryPrediction:
    residual: np.ndarray
    violation_residual: float
    trust: float
    uncertainty: float
    predicted_violation_risk: float
    shield_active: float
    stale_score: float


NOMINAL = PhysParams(0.12, 0.65)
ROBUST_BRANCHES = [PhysParams(0.07, 0.20), NOMINAL, PhysParams(0.30, 1.18)]
PUCK_RADIUS = 0.045
OBSTACLE_RADIUS = 0.055
SUCCESS_RADIUS = 0.075
SAFETY_MARGIN = PUCK_RADIUS + OBSTACLE_RADIUS + 0.006
MODEL_CACHE: dict[PhysParams, mujoco.MjModel] = {}

MAIN_METHODS = [
    "random_candidate",
    "nominal_mpc",
    "robust_worst_case_mpc",
    "last_repair_memory",
    "global_average_repair",
    "knn_repair_memory_v4",
    "online_ridge_residual",
    "corm_repair_memory_v5",
    "corm_no_safety",
    "corm_no_uncertainty",
    "oracle_hidden_deployment",
]
ABLATION_METHODS = [
    "corm_repair_memory_v5",
    "corm_no_safety",
    "corm_no_uncertainty",
    "corm_no_context",
    "corm_limited_memory",
    "global_average_repair",
    "last_repair_memory",
    "knn_repair_memory_v4",
    "nominal_mpc",
    "robust_worst_case_mpc",
    "oracle_hidden_deployment",
]
SPLITS = {
    "nominal": {
        "mass": (0.10, 0.16),
        "friction": (0.50, 0.85),
        "gain": (0.95, 1.05),
        "angle": (-0.02, 0.02),
        "lateral": (-0.004, 0.004),
        "obstacle": 0.015,
    },
    "low_friction_shift": {
        "mass": (0.10, 0.16),
        "friction": (0.12, 0.28),
        "gain": (0.82, 0.98),
        "angle": (-0.06, 0.06),
        "lateral": (-0.012, 0.012),
        "obstacle": 0.026,
    },
    "high_friction_shift": {
        "mass": (0.10, 0.16),
        "friction": (0.95, 1.35),
        "gain": (0.90, 1.08),
        "angle": (-0.08, 0.08),
        "lateral": (-0.012, 0.012),
        "obstacle": 0.026,
    },
    "heavy_object_shift": {
        "mass": (0.24, 0.40),
        "friction": (0.45, 0.90),
        "gain": (0.76, 0.96),
        "angle": (-0.06, 0.06),
        "lateral": (-0.012, 0.012),
        "obstacle": 0.030,
    },
    "light_object_shift": {
        "mass": (0.045, 0.075),
        "friction": (0.35, 0.75),
        "gain": (1.04, 1.30),
        "angle": (-0.08, 0.08),
        "lateral": (-0.014, 0.014),
        "obstacle": 0.030,
    },
    "actuation_bias_shift": {
        "mass": (0.10, 0.18),
        "friction": (0.45, 0.85),
        "gain": (0.70, 1.22),
        "angle": (-0.17, 0.17),
        "lateral": (-0.024, 0.024),
        "obstacle": 0.038,
    },
    "obstacle_shift": {
        "mass": (0.09, 0.18),
        "friction": (0.35, 0.95),
        "gain": (0.86, 1.14),
        "angle": (-0.09, 0.09),
        "lateral": (-0.020, 0.020),
        "obstacle": 0.070,
    },
    "nonstationary_deployment_shift": {
        "mass": (0.07, 0.35),
        "friction": (0.16, 1.25),
        "gain": (0.70, 1.26),
        "angle": (-0.18, 0.18),
        "lateral": (-0.026, 0.026),
        "obstacle": 0.050,
    },
    "combined_shift": {
        "mass": (0.06, 0.42),
        "friction": (0.12, 1.35),
        "gain": (0.68, 1.24),
        "angle": (-0.19, 0.19),
        "lateral": (-0.028, 0.028),
        "obstacle": 0.060,
    },
}
DEFAULT_ABLATION_SPLITS = ["combined_shift", "nonstationary_deployment_shift"]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def make_model(params: PhysParams) -> mujoco.MjModel:
    cached = MODEL_CACHE.get(params)
    if cached is not None:
        return cached
    xml = f"""
    <mujoco model="repair_memory_push_v5">
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


def action_path(
    puck_xy: np.ndarray,
    action: PushAction,
    deployment: Deployment | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    violation = float(min_obstacle_dist < SAFETY_MARGIN)
    return {
        "final_xy": final_xy,
        "violation": violation,
        "effort": effort,
        "min_obstacle_dist": min_obstacle_dist,
    }


def split_rng(split: str, seed: int, phase: int = 0) -> random.Random:
    return random.Random(6300001 + 1597 * seed + 70001 * phase + sum(ord(c) for c in split))


def sample_deployment(split: str, seed: int, phase: int = 0) -> Deployment:
    rng = split_rng(split, seed, phase)
    cfg = SPLITS[split]
    obstacle_mag = cfg["obstacle"]
    return Deployment(
        params=PhysParams(rng.uniform(*cfg["mass"]), rng.uniform(*cfg["friction"])),
        distance_gain=rng.uniform(*cfg["gain"]),
        angle_bias=rng.uniform(*cfg["angle"]),
        lateral_bias=rng.uniform(*cfg["lateral"]),
        obstacle_shift=(rng.uniform(-obstacle_mag, obstacle_mag), rng.uniform(-obstacle_mag, obstacle_mag)),
    )


def deployment_for_episode(split: str, seed: int, episode: int, episodes: int) -> tuple[Deployment, int]:
    if split != "nonstationary_deployment_shift":
        return sample_deployment(split, seed, 0), 0
    phase = int(episode >= max(1, episodes // 2))
    return sample_deployment(split, seed, phase), phase


def sample_task(split: str, seed: int, episode: int) -> TaskSpec:
    rng = random.Random(6309103 + 100003 * seed + 7919 * episode + sum(ord(c) for c in split))
    puck = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)], dtype=float)
    target_angle = rng.uniform(-0.72, 0.72)
    target_radius = rng.uniform(0.27, 0.45)
    target = puck + target_radius * np.array([math.cos(target_angle), math.sin(target_angle)], dtype=float)
    midpoint = 0.50 * (puck + target)
    normal = np.array([-math.sin(target_angle), math.cos(target_angle)], dtype=float)
    if split == "obstacle_shift":
        lateral = rng.choice([-1, 1]) * rng.uniform(0.030, 0.105)
    else:
        lateral = rng.choice([-1, 1]) * rng.uniform(0.050, 0.135)
    obstacle = midpoint + lateral * normal
    return TaskSpec(split, tuple(puck), tuple(target), tuple(obstacle))


def candidate_actions(puck_xy: np.ndarray, target_xy: np.ndarray) -> list[PushAction]:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    remaining = float(np.linalg.norm(target_xy - puck_xy))
    actions: list[PushAction] = []
    for deg in [-50, -30, -15, 0, 15, 30, 50]:
        for offset in [-0.035, 0.0, 0.035]:
            for scale in [0.78, 1.03, 1.28]:
                actions.append(PushAction(base + math.radians(deg), offset, max(0.15, min(0.58, scale * remaining))))
    return actions


def line_clearance(start: np.ndarray, end: np.ndarray, obstacle: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment)) + 1e-8
    t = max(0.0, min(1.0, float(np.dot(obstacle - start, segment) / denom)))
    closest = start + t * segment
    return float(np.linalg.norm(closest - obstacle))


def energy(final_xy: np.ndarray, target_xy: np.ndarray, violation: float, effort: float) -> float:
    return float(np.linalg.norm(final_xy - target_xy)) + 0.30 * float(violation) + 0.03 * effort


def action_features(
    puck_xy: np.ndarray,
    target_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    action: PushAction,
    nominal: dict,
) -> np.ndarray:
    base = math.atan2(float(target_xy[1] - puck_xy[1]), float(target_xy[0] - puck_xy[0]))
    angle_rel = math.atan2(math.sin(action.angle - base), math.cos(action.angle - base))
    direction = np.array([math.cos(action.angle), math.sin(action.angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    geom_end = puck_xy + action.distance * direction + action.offset * normal
    nominal_final = nominal["final_xy"]
    nominal_disp = nominal_final - puck_xy
    nominal_error = target_xy - nominal_final
    init_vec = target_xy - puck_xy
    init_dist = float(np.linalg.norm(init_vec))
    clearance = line_clearance(puck_xy, geom_end, obstacle_xy)
    return np.array(
        [
            1.0,
            math.sin(angle_rel),
            math.cos(angle_rel),
            action.distance,
            action.offset,
            init_dist,
            float(init_vec[0]),
            float(init_vec[1]),
            float(nominal_disp[0]),
            float(nominal_disp[1]),
            float(nominal_error[0]),
            float(nominal_error[1]),
            float(np.linalg.norm(target_xy - geom_end)),
            clearance,
            float(clearance - SAFETY_MARGIN),
            float(np.linalg.norm(obstacle_xy - puck_xy)),
            float(np.linalg.norm(obstacle_xy - target_xy)),
            float(nominal["violation"]),
            float(nominal["min_obstacle_dist"] - SAFETY_MARGIN),
            float(nominal["effort"]),
        ],
        dtype=np.float64,
    )


def prepare_candidates(task: TaskSpec, deployment: Deployment) -> list[dict]:
    puck = np.array(task.puck, dtype=float)
    target = np.array(task.target, dtype=float)
    obstacle = np.array(task.obstacle, dtype=float)
    actions = candidate_actions(puck, target)
    rows: list[dict] = []
    for idx, action in enumerate(actions):
        nominal = rollout_push(NOMINAL, puck, obstacle, action, None)
        true = rollout_push(deployment.params, puck, obstacle, action, deployment)
        robust_scores = []
        for branch in ROBUST_BRANCHES:
            pred = rollout_push(branch, puck, obstacle, action, None)
            robust_scores.append(energy(pred["final_xy"], target, pred["violation"], pred["effort"]))
        feat = action_features(puck, target, obstacle, action, nominal)
        rows.append(
            {
                "action_idx": idx,
                "action": action,
                "feature": feat,
                "target_xy": target,
                "puck_xy": puck,
                "obstacle_xy": obstacle,
                "nominal": nominal,
                "true": true,
                "robust_score": max(robust_scores),
                "nominal_energy": energy(nominal["final_xy"], target, nominal["violation"], nominal["effort"]),
                "true_energy": energy(true["final_xy"], target, true["violation"], true["effort"]),
                "nominal_clearance": float(nominal["min_obstacle_dist"] - SAFETY_MARGIN),
            }
        )
    return rows


class KNNRepairMemory:
    def __init__(self, mode: str, capacity: int = 256, use_context: bool = True) -> None:
        self.mode = mode
        self.capacity = capacity
        self.use_context = use_context
        self.features: list[np.ndarray] = []
        self.residuals: list[np.ndarray] = []
        self.violation_residuals: list[float] = []
        self.last_prediction_error = 0.0

    def reset(self) -> None:
        self.features.clear()
        self.residuals.clear()
        self.violation_residuals.clear()
        self.last_prediction_error = 0.0

    def add(self, feature: np.ndarray, residual: np.ndarray, violation_residual: float) -> None:
        pred = self.predict(feature)
        self.last_prediction_error = float(np.linalg.norm(pred.residual - residual))
        feat = feature if self.use_context else np.zeros_like(feature)
        self.features.append(feat.astype(np.float64))
        self.residuals.append(residual.astype(np.float64))
        self.violation_residuals.append(float(violation_residual))
        if len(self.features) > self.capacity:
            self.features = self.features[-self.capacity :]
            self.residuals = self.residuals[-self.capacity :]
            self.violation_residuals = self.violation_residuals[-self.capacity :]

    def predict(self, feature: np.ndarray) -> MemoryPrediction:
        if not self.features:
            return MemoryPrediction(np.zeros(2, dtype=float), 0.0, 0.0, 0.25, 0.0, 0.0, 0.0)
        if self.mode == "last":
            residual = np.array(self.residuals[-1], dtype=float)
            violation = self.violation_residuals[-1]
            trust = min(0.60, len(self.features) / 18.0)
            return MemoryPrediction(trust * residual, trust * violation, trust, 0.16, 0.0, 0.0, 0.0)
        if self.mode == "global":
            residual = np.mean(np.stack(self.residuals), axis=0)
            violation = float(np.mean(self.violation_residuals))
            spread = float(np.mean(np.linalg.norm(np.stack(self.residuals) - residual[None, :], axis=1)))
            trust = min(0.70, len(self.features) / 20.0) * math.exp(-3.0 * spread)
            return MemoryPrediction(trust * residual, trust * violation, trust, min(0.40, spread + 0.08), 0.0, 0.0, 0.0)
        if len(self.features) < 4:
            return MemoryPrediction(np.zeros(2, dtype=float), 0.0, 0.0, 0.22, 0.0, 0.0, 0.0)
        feat = feature if self.use_context else np.zeros_like(feature)
        feats = np.stack(self.features)
        scale = feats.std(axis=0) + 0.08
        d = np.linalg.norm((feats - feat) / scale, axis=1)
        k = min(7, len(d))
        idx = np.argsort(d)[:k]
        weights = np.exp(-0.85 * d[idx])
        weights /= max(float(weights.sum()), 1e-8)
        residual = np.sum(np.stack([self.residuals[i] for i in idx]) * weights[:, None], axis=0)
        violation = float(np.sum(np.array([self.violation_residuals[i] for i in idx]) * weights))
        local_spread = float(np.sum(np.array([np.linalg.norm(self.residuals[i] - residual) for i in idx]) * weights))
        trust = min(0.78, len(self.features) / 22.0) * math.exp(-0.10 * float(d[idx[0]])) * math.exp(-2.0 * local_spread)
        uncertainty = min(0.50, 0.08 + local_spread + 0.02 * float(d[idx[0]]))
        return MemoryPrediction(trust * residual, trust * violation, trust, uncertainty, 0.0, 0.0, 0.0)


class RidgeRepairMemory:
    def __init__(
        self,
        *,
        capacity: int = 160,
        ridge: float = 0.08,
        use_context: bool = True,
        calibrated: bool = True,
        safety: bool = True,
        uncertainty: bool = True,
    ) -> None:
        self.capacity = capacity
        self.ridge = ridge
        self.use_context = use_context
        self.calibrated = calibrated
        self.safety = safety
        self.uncertainty = uncertainty
        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.beta: np.ndarray | None = None
        self.inv_xtx: np.ndarray | None = None
        self.residual_rmse = 0.12
        self.violation_rmse = 0.25
        self.shock = 0.0
        self.last_prediction_error = 0.0

    def reset(self) -> None:
        self.features.clear()
        self.targets.clear()
        self.beta = None
        self.inv_xtx = None
        self.residual_rmse = 0.12
        self.violation_rmse = 0.25
        self.shock = 0.0
        self.last_prediction_error = 0.0

    def _clean_feature(self, feature: np.ndarray) -> np.ndarray:
        feat = np.array(feature, dtype=np.float64)
        if not self.use_context:
            keep = np.zeros_like(feat)
            keep[0] = 1.0
            keep[1:6] = feat[1:6]
            keep[8:13] = feat[8:13]
            feat = keep
        return feat

    def _fit(self) -> None:
        if not self.features:
            self.beta = None
            self.inv_xtx = None
            return
        x = np.stack(self.features)
        y = np.stack(self.targets)
        reg = self.ridge * np.eye(x.shape[1])
        reg[0, 0] = self.ridge * 0.05
        xtx = x.T @ x + reg
        self.inv_xtx = np.linalg.pinv(xtx)
        self.beta = self.inv_xtx @ x.T @ y
        preds = x @ self.beta
        errors = y - preds
        self.residual_rmse = float(np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1)))) if len(errors) else 0.12
        self.violation_rmse = float(np.sqrt(np.mean(errors[:, 2] ** 2))) if len(errors) else 0.25

    def predict(self, feature: np.ndarray, nominal_violation: float = 0.0, nominal_clearance: float = 0.20) -> MemoryPrediction:
        feat = self._clean_feature(feature)
        if self.beta is None or len(self.features) < 3:
            clearance_risk = max(0.0, min(1.0, (SAFETY_MARGIN - (nominal_clearance + SAFETY_MARGIN)) / 0.05))
            return MemoryPrediction(np.zeros(2), 0.0, 0.0, 0.28, float(nominal_violation + clearance_risk), 0.0, self.shock)
        raw = feat @ self.beta
        leverage = 0.0
        if self.inv_xtx is not None:
            leverage = max(0.0, float(feat @ self.inv_xtx @ feat.T))
        uncertainty = min(0.65, self.residual_rmse + 0.035 * math.sqrt(leverage) + 0.16 / math.sqrt(len(self.features)))
        if not self.uncertainty:
            trust = min(0.92, len(self.features) / 8.0)
        else:
            ramp = 1.0 - math.exp(-len(self.features) / 8.0)
            stability = math.exp(-3.2 * uncertainty)
            shock_penalty = 1.0 - min(0.90, self.shock)
            trust = min(0.88, 0.92 * ramp * stability * shock_penalty)
        violation_raw = float(raw[2])
        clearance_penalty = max(0.0, min(1.0, -nominal_clearance / 0.055))
        risk = max(0.0, min(1.0, float(nominal_violation) + trust * violation_raw + (0.45 * uncertainty if self.uncertainty else 0.0) + 0.35 * clearance_penalty))
        # The shield is intentionally selective: development runs showed that
        # low-confidence continuous penalties can be worse than honest no-shield
        # residual control.  We therefore activate only on high predicted risk.
        shield_active = float(self.safety and risk > 0.40)
        return MemoryPrediction(
            residual=trust * np.array(raw[:2], dtype=float),
            violation_residual=trust * violation_raw,
            trust=float(trust),
            uncertainty=float(uncertainty),
            predicted_violation_risk=float(risk),
            shield_active=shield_active,
            stale_score=float(self.shock),
        )

    def add(self, feature: np.ndarray, residual: np.ndarray, violation_residual: float) -> None:
        pred = self.predict(feature)
        prediction_error = float(np.linalg.norm(pred.residual - residual))
        self.last_prediction_error = prediction_error
        if self.calibrated:
            threshold = max(0.080, 2.1 * max(pred.uncertainty, self.residual_rmse))
            if prediction_error > threshold:
                self.shock = min(1.0, 0.55 + 4.0 * (prediction_error - threshold))
            else:
                self.shock = max(0.0, 0.80 * self.shock - 0.035)
        else:
            self.shock = max(0.0, 0.90 * self.shock)
        feat = self._clean_feature(feature)
        target = np.array([float(residual[0]), float(residual[1]), float(violation_residual)], dtype=np.float64)
        self.features.append(feat)
        self.targets.append(target)
        if len(self.features) > self.capacity:
            self.features = self.features[-self.capacity :]
            self.targets = self.targets[-self.capacity :]
        self._fit()


def make_memory(method: str):
    if method == "last_repair_memory":
        return KNNRepairMemory("last", capacity=1)
    if method == "global_average_repair":
        return KNNRepairMemory("global", capacity=512)
    if method == "knn_repair_memory_v4":
        return KNNRepairMemory("knn", capacity=512, use_context=True)
    if method == "online_ridge_residual":
        return RidgeRepairMemory(capacity=160, calibrated=False, safety=False, uncertainty=False)
    if method == "corm_repair_memory_v5":
        return RidgeRepairMemory(capacity=180, calibrated=True, safety=True, uncertainty=True)
    if method == "corm_no_safety":
        return RidgeRepairMemory(capacity=180, calibrated=True, safety=False, uncertainty=True)
    if method == "corm_no_uncertainty":
        return RidgeRepairMemory(capacity=180, calibrated=True, safety=True, uncertainty=False)
    if method == "corm_no_context":
        return RidgeRepairMemory(capacity=180, calibrated=True, safety=True, uncertainty=True, use_context=False)
    if method == "corm_limited_memory":
        return RidgeRepairMemory(capacity=16, calibrated=True, safety=True, uncertainty=True)
    return None


def repaired_score(row: dict, prediction: MemoryPrediction, use_safety: bool) -> float:
    predicted_final = row["nominal"]["final_xy"] + prediction.residual
    predicted_violation = max(0.0, min(1.0, float(row["nominal"]["violation"]) + prediction.violation_residual))
    base = energy(predicted_final, row["target_xy"], predicted_violation, row["nominal"]["effort"])
    if use_safety and prediction.shield_active > 0.5:
        base += 0.30 * prediction.predicted_violation_risk + 0.08 * float(row["robust_score"])
    return float(base)


def choose_candidate(method: str, rows: list[dict], memory, rng: random.Random) -> tuple[int, MemoryPrediction, float]:
    zero_pred = MemoryPrediction(np.zeros(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if method == "random_candidate":
        chosen = rng.randrange(len(rows))
        return chosen, zero_pred, 0.0
    if method == "nominal_mpc":
        chosen = int(np.argmin([row["nominal_energy"] for row in rows]))
        return chosen, zero_pred, 0.0
    if method == "robust_worst_case_mpc":
        chosen = int(np.argmin([row["robust_score"] for row in rows]))
        return chosen, zero_pred, 0.0
    if method == "oracle_hidden_deployment":
        chosen = int(np.argmin([row["true_energy"] for row in rows]))
        return chosen, zero_pred, 0.0

    use_safety = method in {
        "corm_repair_memory_v5",
        "corm_no_uncertainty",
        "corm_no_context",
        "corm_limited_memory",
    }
    scores: list[float] = []
    predictions: list[MemoryPrediction] = []
    for row in rows:
        if memory is None:
            prediction = zero_pred
        elif isinstance(memory, RidgeRepairMemory):
            prediction = memory.predict(
                row["feature"],
                nominal_violation=float(row["nominal"]["violation"]),
                nominal_clearance=float(row["nominal_clearance"]),
            )
        else:
            prediction = memory.predict(row["feature"])
            risk = max(0.0, min(1.0, float(row["nominal"]["violation"]) + prediction.violation_residual))
            prediction = MemoryPrediction(
                prediction.residual,
                prediction.violation_residual,
                prediction.trust,
                prediction.uncertainty,
                risk,
                0.0,
                prediction.stale_score,
            )
        predictions.append(prediction)
        scores.append(repaired_score(row, prediction, use_safety=use_safety))
    chosen = int(np.argmin(scores))
    return chosen, predictions[chosen], scores[chosen]


def phase_label(episode: int, episodes: int) -> str:
    if episode < episodes / 3:
        return "early"
    if episode < 2 * episodes / 3:
        return "middle"
    return "late"


def run_method_set(split: str, seed: int, episodes: int, methods: Sequence[str], ablation: bool = False) -> list[dict]:
    memories = {method: make_memory(method) for method in methods}
    rngs = {
        method: random.Random(6389 + 99991 * seed + sum(ord(c) for c in split) + sum(ord(c) for c in method))
        for method in methods
    }
    rows: list[dict] = []
    for episode in range(episodes):
        deployment, deployment_phase = deployment_for_episode(split, seed, episode, episodes)
        task = sample_task(split, seed, episode)
        candidates = prepare_candidates(task, deployment)
        target = np.array(task.target, dtype=float)
        puck = np.array(task.puck, dtype=float)
        oracle_energy = min(float(row["true_energy"]) for row in candidates)
        for method in methods:
            memory = memories[method]
            chosen, prediction, predicted_score = choose_candidate(method, candidates, memory, rngs[method])
            selected = candidates[chosen]
            true_out = selected["true"]
            nominal_out = selected["nominal"]
            residual = true_out["final_xy"] - nominal_out["final_xy"]
            violation_residual = float(true_out["violation"] - nominal_out["violation"])
            predicted_final = nominal_out["final_xy"] + prediction.residual
            prediction_error = float(np.linalg.norm(predicted_final - true_out["final_xy"]))
            final_distance = float(np.linalg.norm(true_out["final_xy"] - target))
            if memory is not None:
                memory.add(selected["feature"], residual, violation_residual)
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "phase": phase_label(episode, episodes),
                    "deployment_phase": deployment_phase,
                    "split": split,
                    "method": method,
                    "true_mass": deployment.params.mass,
                    "true_friction": deployment.params.friction,
                    "distance_gain": deployment.distance_gain,
                    "angle_bias": deployment.angle_bias,
                    "lateral_bias": deployment.lateral_bias,
                    "initial_distance": float(np.linalg.norm(target - puck)),
                    "candidate_count": len(candidates),
                    "chosen_action": chosen,
                    "success": float(final_distance <= SUCCESS_RADIUS and true_out["violation"] < 0.5),
                    "final_distance": final_distance,
                    "violation": float(true_out["violation"]),
                    "energy": float(selected["true_energy"]),
                    "oracle_energy": oracle_energy,
                    "energy_regret": float(selected["true_energy"] - oracle_energy),
                    "decision_trust": float(prediction.trust),
                    "decision_uncertainty": float(prediction.uncertainty),
                    "predicted_violation_risk": float(prediction.predicted_violation_risk),
                    "shield_active": float(prediction.shield_active),
                    "stale_score": float(prediction.stale_score),
                    "repair_prediction_error": prediction_error,
                    "predicted_score": float(predicted_score),
                    "ablation": ablation,
                }
            )
    return rows


def ci95(vals: Iterable[float]) -> float:
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / math.sqrt(len(vals))


def bootstrap_ci(vals: Sequence[float], rng_seed: int = 123, reps: int = 1000) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(rng_seed)
    arr = np.array(vals, dtype=float)
    means = []
    for _ in range(reps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalue(vals: Sequence[float], rng_seed: int = 321, reps: int = 4096) -> float:
    if not vals:
        return 1.0
    arr = np.array(vals, dtype=float)
    observed = float(arr.mean())
    if abs(observed) < 1e-12:
        return 1.0
    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(reps):
        signs = rng.choice([-1.0, 1.0], size=len(arr), replace=True)
        if abs(float((arr * signs).mean())) >= abs(observed):
            count += 1
    return float((count + 1) / (reps + 1))


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
        trusts = [float(v["decision_trust"]) for v in vals]
        uncertainties = [float(v["decision_uncertainty"]) for v in vals]
        shields = [float(v["shield_active"]) for v in vals]
        pred_errors = [float(v["repair_prediction_error"]) for v in vals]
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
                "decision_trust_mean": mean(trusts),
                "decision_uncertainty_mean": mean(uncertainties),
                "shield_activation_rate": mean(shields),
                "repair_prediction_error_mean": mean(pred_errors),
            }
        )
        out.append(summary)
    return out


def paired_stats(rows: list[dict]) -> list[dict]:
    proposed = "corm_repair_memory_v5"
    baselines = [method for method in MAIN_METHODS if method != proposed]
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
            regret_lo, regret_hi = bootstrap_ci(regret_delta, rng_seed=637 + len(out))
            success_lo, success_hi = bootstrap_ci(success_delta, rng_seed=937 + len(out))
            out.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_episodes": len(paired),
                    "success_delta_mean": mean(success_delta),
                    "success_delta_boot_lo": success_lo,
                    "success_delta_boot_hi": success_hi,
                    "success_delta_ci95": ci95(success_delta),
                    "success_signflip_p": sign_flip_pvalue(success_delta, rng_seed=1301 + len(out)),
                    "regret_improvement_mean": mean(regret_delta),
                    "regret_improvement_boot_lo": regret_lo,
                    "regret_improvement_boot_hi": regret_hi,
                    "regret_improvement_ci95": ci95(regret_delta),
                    "regret_signflip_p": sign_flip_pvalue(regret_delta, rng_seed=1709 + len(out)),
                    "violation_delta_mean": mean(violation_delta),
                    "violation_delta_ci95": ci95(violation_delta),
                }
            )
    return out


def decision_from_metrics(main_summary: list[dict], pairwise: list[dict]) -> tuple[str, list[str]]:
    aggregate = summarize_from_summary(main_summary, "method")
    by_method = {row["method"]: row for row in aggregate}
    corm = by_method.get("corm_repair_memory_v5")
    reasons: list[str] = []
    if corm is None:
        return "KILL_ARCHIVE", ["CORM summary missing."]
    required = [
        "nominal_mpc",
        "robust_worst_case_mpc",
        "last_repair_memory",
        "global_average_repair",
        "knn_repair_memory_v4",
        "online_ridge_residual",
    ]
    success_gate = True
    regret_gate = True
    safety_gate = True
    for baseline in required:
        base = by_method.get(baseline)
        if base is None:
            success_gate = False
            regret_gate = False
            reasons.append(f"Missing baseline {baseline}.")
            continue
        success_delta = float(corm["success_rate"]) - float(base["success_rate"])
        regret_delta = float(base["energy_regret_mean"]) - float(corm["energy_regret_mean"])
        violation_delta = float(corm["violation_rate"]) - float(base["violation_rate"])
        reasons.append(
            f"aggregate vs {baseline}: success_delta={success_delta:.4f}, "
            f"regret_improvement={regret_delta:.4f}, violation_delta={violation_delta:.4f}"
        )
        success_gate = success_gate and success_delta >= -0.005
        regret_gate = regret_gate and regret_delta >= -0.003
        if baseline in {"nominal_mpc", "robust_worst_case_mpc", "knn_repair_memory_v4"}:
            safety_gate = safety_gate and violation_delta <= 0.015
    stress_pairs = [
        row
        for row in pairwise
        if row["split"] in {"combined_shift", "nonstationary_deployment_shift"}
        and row["baseline"] in {"nominal_mpc", "robust_worst_case_mpc", "knn_repair_memory_v4"}
    ]
    for row in stress_pairs:
        if float(row["violation_delta_mean"]) > 0.03:
            safety_gate = False
            reasons.append(
                f"stress safety failure on {row['split']} vs {row['baseline']}: "
                f"violation_delta={float(row['violation_delta_mean']):.4f}"
            )
    if success_gate and regret_gate and safety_gate:
        return "STRONG_REVISE", reasons
    return "KILL_ARCHIVE", reasons


def summarize_from_summary(summary_rows: list[dict], key: str) -> list[dict]:
    expanded: list[dict] = []
    for row in summary_rows:
        repeats = int(row.get("episodes", 1))
        # Weighted aggregation without keeping raw episode identities.
        expanded.append({**row, "_weight": repeats})
    groups: dict[str, list[dict]] = {}
    for row in expanded:
        groups.setdefault(str(row[key]), []).append(row)
    out: list[dict] = []
    for group_key, vals in sorted(groups.items()):
        total = sum(float(v["_weight"]) for v in vals)
        entry = {key: group_key, "episodes": int(total)}
        for metric in [
            "success_rate",
            "final_distance_mean",
            "violation_rate",
            "energy_regret_mean",
            "decision_trust_mean",
            "decision_uncertainty_mean",
            "shield_activation_rate",
            "repair_prediction_error_mean",
        ]:
            entry[metric] = sum(float(v[metric]) * float(v["_weight"]) for v in vals) / max(total, 1.0)
        out.append(entry)
    return out


def learning_curve(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["phase", "split", "method"])


def calibration_summary(rows: list[dict]) -> list[dict]:
    repair_methods = {
        "last_repair_memory",
        "global_average_repair",
        "knn_repair_memory_v4",
        "online_ridge_residual",
        "corm_repair_memory_v5",
        "corm_no_safety",
        "corm_no_uncertainty",
        "corm_no_context",
        "corm_limited_memory",
    }
    return summarize([row for row in rows if row["method"] in repair_methods], ["split", "phase", "method"])


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
                clean[key] = f"{value:.6f}"
        formatted.append(clean)
    return formatted


def plot_results(metrics: list[dict], learning: list[dict], ablation: list[dict], calibration: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    splits = sorted({row["split"] for row in metrics})
    methods = [
        "nominal_mpc",
        "robust_worst_case_mpc",
        "global_average_repair",
        "knn_repair_memory_v4",
        "online_ridge_residual",
        "corm_repair_memory_v5",
        "oracle_hidden_deployment",
    ]
    labels = ["Nominal", "Robust", "Global", "kNN v4", "Ridge", "CORM", "Oracle"]
    x = np.arange(len(splits))
    width = 0.11
    plt.figure(figsize=(13, 5.2))
    for idx, method in enumerate(methods):
        vals = []
        for split in splits:
            match = [row for row in metrics if row["split"] == split and row["method"] == method]
            vals.append(float(match[0]["success_rate"]) if match else 0.0)
        plt.bar(x + (idx - 3) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=24, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.02)
    plt.title("Repair-memory methods across frozen deployment shifts")
    plt.legend(ncol=7, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_success_by_split.png", dpi=180)
    plt.close()

    phases = ["early", "middle", "late"]
    plt.figure(figsize=(8.2, 5.0))
    for method, label in [
        ("nominal_mpc", "Nominal"),
        ("robust_worst_case_mpc", "Robust"),
        ("knn_repair_memory_v4", "kNN v4"),
        ("online_ridge_residual", "Ridge"),
        ("corm_repair_memory_v5", "CORM"),
    ]:
        vals = []
        for phase in phases:
            phase_vals = [float(row["success_rate"]) for row in learning if row["phase"] == phase and row["method"] == method]
            vals.append(mean(phase_vals) if phase_vals else 0.0)
        plt.plot(phases, vals, marker="o", label=label)
    plt.ylabel("Average success across splits")
    plt.title("Repeated-deployment learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_learning_curve.png", dpi=180)
    plt.close()

    order = sorted(ablation, key=lambda row: float(row["energy_regret_mean"]))
    plt.figure(figsize=(9.5, 5.2))
    plt.barh([f"{row['split']}:{row['method']}" for row in order], [float(row["energy_regret_mean"]) for row in order])
    plt.xlabel("Energy regret vs hidden-deployment oracle")
    plt.title("Ablations on combined and nonstationary shifts")
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_ablation_regret.png", dpi=180)
    plt.close()

    corm = [row for row in calibration if row["method"] == "corm_repair_memory_v5"]
    plt.figure(figsize=(8.2, 4.8))
    trust_vals = []
    err_vals = []
    for phase in phases:
        trust_phase = [float(row["decision_trust_mean"]) for row in corm if row["phase"] == phase]
        err_phase = [float(row["repair_prediction_error_mean"]) for row in corm if row["phase"] == phase]
        trust_vals.append(mean(trust_phase) if trust_phase else 0.0)
        err_vals.append(mean(err_phase) if err_phase else 0.0)
    plt.plot(phases, trust_vals, marker="o", label="Trust")
    plt.plot(phases, err_vals, marker="s", label="Prediction error")
    plt.ylabel("Mean value")
    plt.title("CORM trust calibration diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "repair_trust_calibration.png", dpi=180)
    plt.close()


def write_summary_txt(
    args: argparse.Namespace,
    main_summary: list[dict],
    aggregate: list[dict],
    pairwise: list[dict],
    decision: str,
    decision_reasons: list[str],
) -> None:
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("MuJoCo repeated-deployment repair-memory benchmark for Paper 63, v5\n")
        f.write(
            f"seeds={args.seeds} episodes_per_seed_split_method={args.episodes} "
            f"splits={','.join(args.splits)} methods={','.join(args.methods)}\n"
        )
        f.write(f"terminal_decision={decision}\n\n")
        f.write("Aggregate method summary:\n")
        for row in aggregate:
            f.write(
                f"{row['method']} success={float(row['success_rate']):.4f} "
                f"regret={float(row['energy_regret_mean']):.4f} "
                f"violation={float(row['violation_rate']):.4f} "
                f"trust={float(row['decision_trust_mean']):.4f}\n"
            )
        f.write("\nCORM split summary:\n")
        for row in main_summary:
            if row["method"] == "corm_repair_memory_v5":
                f.write(
                    f"{row['split']} corm success={float(row['success_rate']):.4f}+/-{float(row['success_ci95']):.4f} "
                    f"regret={float(row['energy_regret_mean']):.4f}+/-{float(row['energy_regret_ci95']):.4f} "
                    f"violation={float(row['violation_rate']):.4f} trust={float(row['decision_trust_mean']):.4f}\n"
                )
        f.write("\nDecision audit:\n")
        for reason in decision_reasons:
            f.write(f"- {reason}\n")
        f.write("\nPairwise stress rows:\n")
        for row in pairwise:
            if row["split"] in {"combined_shift", "nonstationary_deployment_shift"}:
                f.write(
                    f"{row['split']} vs {row['baseline']} success_delta={float(row['success_delta_mean']):.4f} "
                    f"regret_improvement={float(row['regret_improvement_mean']):.4f} "
                    f"violation_delta={float(row['violation_delta_mean']):.4f}\n"
                )


def run(args: argparse.Namespace) -> None:
    global RESULTS, FIGURES
    RESULTS = Path(args.results_dir).resolve()
    FIGURES = Path(args.figures_dir).resolve()
    ensure_dirs()
    raw_rows: list[dict] = []
    for split in args.splits:
        for seed in range(args.seeds):
            raw_rows.extend(run_method_set(split, seed, args.episodes, args.methods, ablation=False))
            write_rows(RESULTS / "repair_memory_raw.partial.csv", format_rows(raw_rows))
        write_rows(RESULTS / "repair_memory_metrics.partial.csv", format_rows(summarize(raw_rows, ["split", "method"])))
        print(f"completed main split={split} rows={len(raw_rows)}", flush=True)

    ablation_rows: list[dict] = []
    for split in args.ablation_splits:
        for seed in range(args.seeds):
            ablation_rows.extend(run_method_set(split, seed, args.episodes, args.ablation_methods, ablation=True))
            write_rows(RESULTS / "repair_memory_ablation_raw.partial.csv", format_rows(ablation_rows))
        write_rows(RESULTS / "repair_memory_ablation.partial.csv", format_rows(summarize(ablation_rows, ["split", "method"])))
        print(f"completed ablation split={split} rows={len(ablation_rows)}", flush=True)

    main_summary = summarize(raw_rows, ["split", "method"])
    seed_summary = summarize(raw_rows, ["split", "method", "seed"])
    ablation_summary = summarize(ablation_rows, ["split", "method"])
    learning = learning_curve(raw_rows)
    calibration = calibration_summary(raw_rows + ablation_rows)
    pairwise = paired_stats(raw_rows)
    aggregate = summarize_from_summary(main_summary, "method")
    decision, decision_reasons = decision_from_metrics(main_summary, pairwise)

    write_rows(RESULTS / "repair_memory_raw.csv", format_rows(raw_rows))
    write_rows(RESULTS / "repair_memory_ablation_raw.csv", format_rows(ablation_rows))
    write_rows(RESULTS / "repair_memory_metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "repair_memory_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "repair_memory_ablation.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "repair_memory_learning_curve.csv", format_rows(learning))
    write_rows(RESULTS / "repair_memory_calibration.csv", format_rows(calibration))
    write_rows(RESULTS / "repair_memory_pairwise.csv", format_rows(pairwise))
    write_rows(RESULTS / "repair_memory_aggregate.csv", format_rows(aggregate))
    write_rows(RESULTS / "metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "raw_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "ablation_metrics.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "stress_sweep.csv", format_rows(learning))
    write_rows(RESULTS / "pairwise_stats.csv", format_rows(pairwise))
    write_rows(RESULTS / "decision_audit.csv", [{"terminal_decision": decision, "reason": r} for r in decision_reasons])
    write_rows(FIGURES / "stress_curve_data.csv", format_rows(learning))
    negative_cases = [
        {
            "case": "nonrecurring_random_damage",
            "observed": "persistent memories can retrieve irrelevant residuals when deployment errors are not recurring",
            "paper_status": "limitation",
        },
        {
            "case": "semantic_goal_error",
            "observed": "physical repair memory cannot correct a wrong task specification or target",
            "paper_status": "limitation",
        },
        {
            "case": "unobserved_change_point",
            "observed": "nonstationary deployment shifts can make stale memory worse than robust or reset baselines",
            "paper_status": "stress_test",
        },
        {
            "case": "custom_mujoco_only",
            "observed": "without hardware or a public benchmark, positive evidence supports strong-revise at best",
            "paper_status": "limitation",
        },
    ]
    write_rows(RESULTS / "negative_cases.csv", negative_cases)
    plot_results(main_summary, learning, ablation_summary, calibration)
    write_summary_txt(args, main_summary, aggregate, pairwise, decision, decision_reasons)
    print(f"terminal_decision={decision}", flush=True)
    print(f"wrote repair-memory benchmark results to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    parser.add_argument("--ablation-splits", nargs="+", default=DEFAULT_ABLATION_SPLITS)
    parser.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    parser.add_argument("--ablation-methods", nargs="+", default=ABLATION_METHODS)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--figures-dir", default=str(FIGURES))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
