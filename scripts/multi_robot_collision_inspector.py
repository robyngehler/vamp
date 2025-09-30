#!/usr/bin/env python3
"""Interactive multi-robot collision inspector using ``vamp.fkcc_multi_all``.

This tool spawns one or more robots in PyBullet, lets you adjust their joint
values live, and repeatedly evaluates ``vamp.fkcc_multi_all`` against the
current configurations.  The collision status is displayed in the PyBullet GUI
so you can verify whether a configuration is collision free while you tweak it.

Robot specifications are supplied with ``--robot``.  Each spec accepts an
optional base-frame translation and Euler XYZ rotation using ``@`` separators:

    panda                      # Panda at the origin (identity base transform)
    panda@0.75,0,0             # Panda translated +0.75 m along X
    gp4@1,0,0@0,0,1.5708       # GP4 translated and rotated 90 degrees about Z

You can pass ``--robot`` multiple times to mix different robots or duplicate
the same one at different base transforms.

Each robot can be assigned a friendly label by prefixing the spec with
``label=``—for example ``left=panda`` or ``right=gp4@0.75,0,0``.  Labels show up
in the PyBullet sliders and are accepted by ``--state`` overrides so you can
copy/paste joint dumps directly from logs:

    ./multi_robot_collision_inspector.py \
        --robot left=panda \
        --robot right=panda@0.75,0,0 \
        --state "left=-1.63 1.93 -0.03 -2.55 0.55 -2.97 0" \
        --state "right=1.45 -1.77 -0.86 2.64 -1.62 2.01 0"
"""

from __future__ import annotations

import argparse
import math
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

import vamp
from vamp.transformations import euler_matrix, quaternion_from_matrix


@dataclass
class RobotSpec:
    name: str
    label: str
    transform: np.ndarray  # 4x4 float32


@dataclass
class RobotRuntime:
    spec: RobotSpec
    module: object
    body_id: int
    joint_names: Sequence[str]
    joint_indices: Sequence[int]
    link_names: Dict[int, str]
    slider_ids: Sequence[int]
    configuration: np.ndarray  # working buffer
    joint_limits: Sequence[Tuple[float, float]]


@dataclass
class ObstacleSpec:
    kind: str
    position: np.ndarray
    size: np.ndarray


def _parse_float_list(text: str) -> List[float]:
    tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
    if not tokens:
        raise ValueError("No numeric values provided")
    return [float(token) for token in tokens]


def _parse_state_override(raw: str) -> Tuple[str, List[float]]:
    if "=" not in raw:
        raise ValueError(f"State override '{raw}' must use the form label=values")
    key, values = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"State override '{raw}' is missing a label")
    return key, _parse_float_list(values)


def _normalize_state_key(key: str) -> str:
    key = key.strip()
    if key.startswith("#"):
        key = key[1:]
    return key.lower()


def _normalize_link_name(name: str) -> str:
    return name.replace("::", "/").strip().lower()


def _normalize_link_pairs(raw: Dict[str, Set[Tuple[str, str]]]) -> Dict[str, Set[Tuple[str, str]]]:
    normalized: Dict[str, Set[Tuple[str, str]]] = {}
    for robot, pairs in raw.items():
        norm_pairs: Set[Tuple[str, str]] = set()
        for first, second in pairs:
            norm_pairs.add(tuple(sorted((_normalize_link_name(first), _normalize_link_name(second)))))
        normalized[robot] = norm_pairs
    return normalized


IGNORED_LINK_COLLISIONS: Dict[str, Set[Tuple[str, str]]] = _normalize_link_pairs(
    {
        "gp4": {
            ("fts", "link_4"),
            ("fts", "link_5"),
            ("fts", "link_6"),
            ("link_4", "link_6"),
        },
        "panda": [
            ("panda_link7", "panda_hand"),
        ]
    }
)


def _should_ignore_link_collision(robot_name: str, link_a: str, link_b: str) -> bool:
    normalized_pairs = IGNORED_LINK_COLLISIONS.get(robot_name)
    if not normalized_pairs:
        return False
    normalized = tuple(sorted((_normalize_link_name(link_a), _normalize_link_name(link_b))))
    return normalized in normalized_pairs


def _sanitize_configuration(values: Sequence[float], dimension: int, label: str) -> np.ndarray:
    sanitized = list(values)
    if len(sanitized) < dimension:
        missing = dimension - len(sanitized)
        print(
            f"[multi_robot_collision_inspector] '{label}' provided {len(sanitized)} joint values; padding "
            f"{missing} trailing value(s) with 0.0",
            file=sys.stderr,
        )
        sanitized.extend([0.0] * missing)
    elif len(sanitized) > dimension:
        print(
            f"[multi_robot_collision_inspector] '{label}' provided {len(sanitized)} joint values; truncating "
            f"to the first {dimension}",
            file=sys.stderr,
        )
        sanitized = sanitized[:dimension]
    return np.asarray(sanitized, dtype=np.float32)


def _parse_obstacle(raw: str) -> ObstacleSpec:
    parts = raw.split("@")
    if not parts:
        raise ValueError("Obstacle specification cannot be empty")

    kind = parts[0].strip().lower()
    if kind != "cube":
        raise ValueError(f"Unsupported obstacle kind '{parts[0]}' (only 'cube' is supported)")

    if len(parts) != 3:
        raise ValueError("Cube obstacle must be specified as cube@x,y,z@lx,ly,lz")

    try:
        position = np.asarray([float(v) for v in parts[1].split(",")], dtype=np.float32)
        size = np.asarray([float(v) for v in parts[2].split(",")], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Failed to parse obstacle floats: {exc}") from exc

    if position.shape != (3,) or size.shape != (3,):
        raise ValueError("Cube obstacle position and size must each provide exactly three floats")

    if np.any(size <= 0.0):
        raise ValueError("Cube obstacle sizes must be strictly positive")

    return ObstacleSpec(kind="cube", position=position, size=size)


def _build_initial_overrides(specs: Sequence[RobotSpec], raw_states: Optional[Sequence[str]]) -> Dict[int, np.ndarray]:
    if not raw_states:
        return {}

    parsed: Dict[str, List[float]] = {}
    for entry in raw_states:
        key, values = _parse_state_override(entry)
        parsed[_normalize_state_key(key)] = values

    overrides: Dict[int, np.ndarray] = {}
    used_keys: Set[str] = set()
    for index, spec in enumerate(specs):
        module = getattr(vamp, spec.name)
        dimension = int(module.dimension()) if hasattr(module, "dimension") else len(module.joint_names())

        candidates = {
            _normalize_state_key(spec.label),
            _normalize_state_key(spec.name),
            _normalize_state_key(str(index)),
            _normalize_state_key(f"{spec.label}_{index}"),
            _normalize_state_key(f"{spec.name}_{index}"),
        }

        matched_values: Optional[List[float]] = None
        matched_key: Optional[str] = None
        for candidate in candidates:
            if candidate in parsed:
                matched_values = parsed[candidate]
                matched_key = candidate
                break

        if matched_values is None:
            continue

        overrides[index] = _sanitize_configuration(matched_values, dimension, spec.label)
        if matched_key:
            used_keys.add(matched_key)

    unused = set(parsed) - used_keys
    if unused:
        print(
            "[multi_robot_collision_inspector] No robot matched state override(s): "
            + ", ".join(sorted(unused)),
            file=sys.stderr,
        )

    return overrides


def _world_spheres(runtime: RobotRuntime) -> List[Tuple[np.ndarray, float]]:
    transform = runtime.spec.transform.astype(np.float32)
    spheres = runtime.module.fk(runtime.configuration.tolist())
    world: List[Tuple[np.ndarray, float]] = []
    for sphere in spheres:
        local = np.array([sphere.x, sphere.y, sphere.z, 1.0], dtype=np.float32)
        position = transform @ local
        world.append((position[:3], float(sphere.r)))
    return world


def _spheres_collide(center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float) -> bool:
    delta = center_a - center_b
    distance_sq = float(np.dot(delta, delta))
    radius_sum = radius_a + radius_b
    return distance_sq <= float(radius_sum * radius_sum)


def _compute_vamp_contacts(
    runtimes: Sequence[RobotRuntime], env: vamp.Environment
) -> Tuple[Set[str], List[str]]:
    summaries: Set[str] = set()
    lines: List[str] = []

    world_spheres = [_world_spheres(runtime) for runtime in runtimes]

    for runtime, spheres_world in zip(runtimes, world_spheres):
        env_hits, self_pairs = runtime.module.debug(runtime.configuration.tolist(), env)

        for sphere_index, hits in enumerate(env_hits):
            if not hits:
                continue
            unique_hits = sorted({str(name) for name in hits})
            key = f"env|{runtime.spec.label}|{sphere_index}|{','.join(unique_hits)}"
            if key not in summaries:
                summaries.add(key)
                lines.append(
                    f"{runtime.spec.label}: sphere {sphere_index} vs environment ({', '.join(unique_hits)})"
                )

        for pair in self_pairs:
            a, b = sorted(pair)
            key = f"self|{runtime.spec.label}|{a}|{b}"
            if key in summaries:
                continue
            summaries.add(key)
            penetration = (
                float(np.linalg.norm(spheres_world[a][0] - spheres_world[b][0]))
                - (spheres_world[a][1] + spheres_world[b][1])
            )
            lines.append(
                f"{runtime.spec.label}: sphere {a} <> sphere {b} (penetration {penetration:.4f} m)"
            )

    for first in range(len(runtimes)):
        for second in range(first + 1, len(runtimes)):
            label_a = runtimes[first].spec.label
            label_b = runtimes[second].spec.label
            for idx_a, (center_a, radius_a) in enumerate(world_spheres[first]):
                for idx_b, (center_b, radius_b) in enumerate(world_spheres[second]):
                    if not _spheres_collide(center_a, radius_a, center_b, radius_b):
                        continue
                    key = f"cross|{label_a}|{idx_a}|{label_b}|{idx_b}"
                    if key in summaries:
                        continue
                    summaries.add(key)
                    penetration = float(np.linalg.norm(center_a - center_b)) - (radius_a + radius_b)
                    lines.append(
                        f"{label_a} sphere {idx_a} <> {label_b} sphere {idx_b} "
                        f"(penetration {penetration:.4f} m)"
                    )

    return summaries, lines


def _collect_pybullet_contacts(client: BulletClient, runtimes: Sequence[RobotRuntime]) -> Tuple[Set[str], List[str]]:
    summaries: Set[str] = set()
    lines: List[str] = []

    for runtime in runtimes:
        label = runtime.spec.label
        for contact in client.getContactPoints(bodyA=runtime.body_id, bodyB=runtime.body_id):
            link_a = int(contact[3])
            link_b = int(contact[4])
            if link_a == link_b:
                continue
            name_a_raw = runtime.link_names.get(link_a, f"link{link_a}")
            name_b_raw = runtime.link_names.get(link_b, f"link{link_b}")
            if _should_ignore_link_collision(runtime.spec.name, name_a_raw, name_b_raw):
                continue
            ordered = tuple(sorted((link_a, link_b)))
            key = f"self|{label}|{ordered[0]}|{ordered[1]}"
            if key in summaries:
                continue
            summaries.add(key)
            name_a = runtime.link_names.get(ordered[0], f"link{ordered[0]}")
            name_b = runtime.link_names.get(ordered[1], f"link{ordered[1]}")
            distance = float(contact[8])
            lines.append(f"{label}: {name_a} <> {name_b} (distance {distance:.4f} m)")

    for first in range(len(runtimes)):
        for second in range(first + 1, len(runtimes)):
            runtime_a = runtimes[first]
            runtime_b = runtimes[second]
            label_a = runtime_a.spec.label
            label_b = runtime_b.spec.label
            for contact in client.getContactPoints(bodyA=runtime_a.body_id, bodyB=runtime_b.body_id):
                link_a = int(contact[3])
                link_b = int(contact[4])
                key = f"cross|{label_a}|{link_a}|{label_b}|{link_b}"
                if key in summaries:
                    continue
                summaries.add(key)
                name_a = runtime_a.link_names.get(link_a, f"link{link_a}")
                name_b = runtime_b.link_names.get(link_b, f"link{link_b}")
                distance = float(contact[8])
                lines.append(
                    f"{label_a} {name_a} <> {label_b} {name_b} (distance {distance:.4f} m)"
                )

    return summaries, lines


def _randomize_and_compare(
    client: BulletClient,
    runtimes: Sequence[RobotRuntime],
    env: vamp.Environment,
    iterations: int,
    seed: Optional[int],
    base_configurations: Sequence[np.ndarray],
) -> None:
    if iterations <= 0 or not runtimes:
        return

    rng = random.Random(seed)
    print(
        f"[random] Evaluating {iterations} random sample(s) (seed={seed if seed is not None else 'auto'})"
    )

    mismatches = 0
    errors = 0

    for sample_idx in range(iterations):
        states = []
        for runtime in runtimes:
            draws: List[float] = []
            for lower, upper in runtime.joint_limits:
                lo = lower if math.isfinite(lower) else -math.pi
                hi = upper if math.isfinite(upper) else math.pi
                if not (lo < hi):
                    lo, hi = -math.pi, math.pi
                draws.append(rng.uniform(lo, hi))

            runtime.configuration[:] = draws
            for joint_index, value in zip(runtime.joint_indices, draws):
                client.resetJointState(runtime.body_id, joint_index, value)

            states.append(
                {
                    "robot": runtime.spec.name,
                    "configuration": draws,
                    "transform": runtime.spec.transform.astype(np.float32),
                }
            )

        try:
            vamp_result = vamp.fkcc_multi_all(states, env)
        except ValueError as exc:
            errors += 1
            vamp_result = False
            print(
                f"[random {sample_idx + 1}/{iterations}] fkcc_multi_all error: {exc}"
            )

        client.performCollisionDetection()
        pb_summary, pb_lines = _collect_pybullet_contacts(client, runtimes)
        vamp_summary, vamp_lines = _compute_vamp_contacts(runtimes, env)

        pb_collision = len(pb_summary) > 0
        summary = (
            f"[random {sample_idx + 1}/{iterations}] VAMP={'collision' if not vamp_result else 'free'} | "
            f"PyBullet={'collision' if pb_collision else 'free'}"
        )

        if (not vamp_result) != pb_collision:
            mismatches += 1
            print(summary + " -> mismatch")
            if pb_lines:
                print("  PyBullet contacts:")
                for line in pb_lines:
                    print(f"    {line}")
            else:
                print("  PyBullet contacts: none")

            if vamp_lines:
                print("  VAMP collisions:")
                for line in vamp_lines:
                    print(f"    {line}")
            else:
                print("  VAMP collisions: none")

            print("  Joint samples:")
            for runtime in runtimes:
                formatted = ", ".join(f"{value:.4f}" for value in runtime.configuration)
                print(f"    {runtime.spec.label}: [{formatted}]")
            print("---")
        else:
            print(summary)

    print(
        f"[random] Completed {iterations} sample(s): {mismatches} mismatch(es), {errors} error(s)."
    )

    for runtime, baseline in zip(runtimes, base_configurations):
        runtime.configuration[:] = baseline
        for joint_index, value in zip(runtime.joint_indices, baseline):
            client.resetJointState(runtime.body_id, joint_index, float(value))

def _place_obstacles(
    client: BulletClient,
    env: vamp.Environment,
    obstacles: Sequence[ObstacleSpec]
) -> None:
    for obstacle in obstacles:
        if obstacle.kind != "cube":
            continue

        half_extents = (obstacle.size * 0.5).astype(np.float32)
        collision = client.createCollisionShape(
            pb.GEOM_BOX, halfExtents=half_extents.tolist()
        )
        visual = client.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half_extents.tolist(),
            rgbaColor=[0.7, 0.7, 0.7, 0.4],
        )
        client.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=obstacle.position.tolist(),
        )

        env.add_cuboid(
            vamp.Cuboid(
                [float(obstacle.position[0]),
                float(obstacle.position[1]),
                float(obstacle.position[2])],
                [0.0, 0.0, 0.0], # euler xyz
                [float(half_extents[0]),
                float(half_extents[1]),
                float(half_extents[2])],
            )
        )



def _unique_paths(paths: Iterable[Path]) -> List[str]:
    seen: List[str] = []
    for path in paths:
        try:
            resolved = str(path.resolve())
        except FileNotFoundError:
            continue
        if resolved not in seen:
            seen.append(resolved)
    return seen


def _default_urdf(name: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "resources" / name / f"{name}_spherized.urdf",
        repo_root / "resources" / name / f"{name}_spherized_visual.urdf",
        repo_root / "resources" / name / f"{name}.urdf",
        repo_root.parent / "foam" / "assets" / f"{name}_lego" / f"{name}_spherized.urdf",
        repo_root.parent / "foam" / "assets" / f"{name}_lego" / f"{name}_spherized_visual.urdf",
        repo_root.parent / "foam" / "assets" / name / f"{name}_spherized_visual.urdf",
        repo_root.parent / "foam" / "assets" / name / f"{name}.urdf",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not locate a URDF for robot '{name}'. Tried: {', '.join(str(c) for c in candidates)}")


def parse_robot_spec(raw: str) -> RobotSpec:
    segments = raw.split("@")
    head = segments[0]

    label: Optional[str] = None
    name_part = head
    if "=" in head:
        alias, name_token = head.split("=", 1)
        alias = alias.strip()
        name_part = name_token.strip()
        if not name_part:
            raise ValueError(f"Invalid robot spec '{raw}': missing robot name after '='")
        if not alias:
            raise ValueError(f"Invalid robot spec '{raw}': alias cannot be empty")
        label = alias

    name = name_part.strip()
    if not name:
        raise ValueError(f"Invalid robot spec '{raw}': missing name")

    if label is None:
        label = name

    translation = np.zeros(3, dtype=np.float32)
    rotation = np.zeros(3, dtype=np.float32)

    if len(segments) >= 2 and segments[1]:
        values = [float(v) for v in segments[1].split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError(f"Robot spec '{raw}' translation must provide 3 comma-separated floats")
        translation = np.asarray(values, dtype=np.float32)

    if len(segments) >= 3 and segments[2]:
        values = [float(v) for v in segments[2].split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError(f"Robot spec '{raw}' rotation must provide 3 comma-separated floats (radians)")
        rotation = np.asarray(values, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = euler_matrix(*rotation)[:3, :3].astype(np.float32)
    transform[:3, 3] = translation

    return RobotSpec(name=name, label=label, transform=transform)


def _load_robot(
    client: BulletClient,
    urdf_path: Path,
    spec: RobotSpec,
    search_paths: Sequence[str],
    ignore_visuals: bool,
    initial_configuration: Optional[Sequence[float]] = None,
) -> RobotRuntime:
    for path in search_paths:
        client.setAdditionalSearchPath(path)

    quat_xyzw = quaternion_from_matrix(spec.transform)
    base_quat = (quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])
    base_pos = spec.transform[:3, 3].tolist()

    flags = pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_USE_SELF_COLLISION
    if ignore_visuals:
        flags |= pb.URDF_IGNORE_VISUAL_SHAPES

    body_id = client.loadURDF(
        str(urdf_path),
        basePosition=base_pos,
        baseOrientation=base_quat,
        useFixedBase=True,
        flags=flags,
    )

    joint_map = {}
    link_names: Dict[int, str] = {-1: "base"}
    joint_indices: List[int] = []
    joint_names = getattr(vamp, spec.name).joint_names()
    for joint_index in range(client.getNumJoints(body_id)):
        info = client.getJointInfo(body_id, joint_index)
        joint_map[info[1].decode()] = joint_index
        link_name = info[12].decode() if isinstance(info[12], (bytes, bytearray)) else str(info[12])
        link_names[joint_index] = link_name

    sliders: List[int] = []
    configuration = np.zeros(len(joint_names), dtype=np.float32)
    limits: List[Tuple[float, float]] = []
    for idx, joint_name in enumerate(joint_names):
        bullet_index = joint_map.get(joint_name)
        if bullet_index is None:
            raise RuntimeError(f"Joint '{joint_name}' not found in URDF '{urdf_path}' for robot '{spec.name}'")
        info = client.getJointInfo(body_id, bullet_index)
        lower, upper = info[8], info[9]
        if not (lower < upper) or math.isinf(lower) or math.isinf(upper):
            lower, upper = -math.pi, math.pi
        default_value = 0.0
        if initial_configuration is not None and idx < len(initial_configuration):
            default_value = float(initial_configuration[idx])
        slider_id = client.addUserDebugParameter(
            f"{spec.label}[{idx}] {joint_name}", lower, upper, default_value
        )
        sliders.append(slider_id)
        joint_indices.append(bullet_index)
        configuration[idx] = default_value
        limits.append((float(lower), float(upper)))
        client.resetJointState(body_id, bullet_index, default_value)

    module = getattr(vamp, spec.name)

    return RobotRuntime(
        spec=spec,
        module=module,
        body_id=body_id,
        joint_names=joint_names,
        joint_indices=joint_indices,
        link_names=link_names,
        slider_ids=sliders,
        configuration=configuration,
        joint_limits=limits,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive multi-robot collision inspector")
    parser.add_argument(
        "--robot",
        action="append",
        dest="robots",
        help="Robot spec name[@tx,ty,tz[@roll,pitch,yaw]] (can be repeated).",
    )
    parser.add_argument(
        "--urdf",
        action="append",
        dest="urdfs",
        help="Optional URDF override path for the corresponding --robot entry.",
    )
    parser.add_argument(
        "--state",
        action="append",
        dest="states",
        metavar="LABEL=VALUES",
        help=(
            "Initial joint values for a robot. Use the robot alias, name, or index (0-based), and supply "
            "comma or space separated numbers. Example: --state \"left_arm=-1.63 1.93 -0.03 -2.55 0.55 -2.97 0\""
        ),
    )
    parser.add_argument("--sleep", type=float, default=1.0 / 120.0, help="Loop sleep in seconds (default: 1/120)")
    parser.add_argument("--ignore-visuals", action="store_true", help="Disable URDF visual meshes (avoids missing texture warnings)")
    parser.add_argument("--no-gui", action="store_true", help="Run without the GUI (sliders disabled; not recommended)")
    parser.add_argument("--print", action="store_true", help="Print collision status changes to stdout")
    parser.add_argument(
        "--show-collisions",
        action="store_true",
        help="Log PyBullet and VAMP collision pairs when they change",
    )
    parser.add_argument(
        "--randomize",
        type=int,
        default=0,
        metavar="N",
        help="Generate N random joint samples per run to compare VAMP vs PyBullet before launching the GUI",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for the random sampler used with --randomize",
    )
    parser.add_argument(
        "--obstacle",
        action="append",
        dest="obstacles",
        metavar="cube@x,y,z@lx,ly,lz",
        help="Add an axis-aligned cuboid obstacle at the given center with edge lengths (meters).",
    )
    return parser


def _status_color(collision_free: bool) -> List[float]:
    return [0.0, 1.0, 0.0] if collision_free else [1.0, 0.1, 0.1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    specs_raw = args.robots or ["panda", "panda@0.75,0,0"]
    specs = [parse_robot_spec(raw) for raw in specs_raw]

    for spec in specs:
        if spec.name not in vamp.robots:
            raise SystemExit(f"Robot '{spec.name}' is not available. Known robots: {', '.join(vamp.robots)}")

    obstacle_specs: List[ObstacleSpec] = []
    if args.obstacles:
        for raw in args.obstacles:
            try:
                obstacle_specs.append(_parse_obstacle(raw))
            except ValueError as exc:
                raise SystemExit(f"Failed to parse --obstacle '{raw}': {exc}") from exc

    initial_overrides = _build_initial_overrides(specs, args.states)

    urdf_overrides: List[Optional[Path]] = []
    overrides = args.urdfs or []
    if overrides and len(overrides) != len(specs):
        raise SystemExit("If --urdf is provided it must be supplied once per --robot entry.")
    for item in overrides:
        urdf_overrides.append(Path(item))
    while len(urdf_overrides) < len(specs):
        urdf_overrides.append(None)

    if args.no_gui:
        raise SystemExit("--no-gui is not supported for the interactive inspector; omit the flag to launch the GUI.")

    connection_mode = pb.GUI

    client = BulletClient(connection_mode=connection_mode)
    client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    client.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
    client.resetDebugVisualizerCamera(
        cameraDistance=2.4,
        cameraYaw=135,
        cameraPitch=-35,
        cameraTargetPosition=[0.5, 0.0, 0.6],
    )

    repo_root = Path(__file__).resolve().parent.parent
    search_roots = _unique_paths([
        repo_root,
        repo_root / "resources",
        repo_root.parent / "foam" / "assets",
    ])

    runtimes: List[RobotRuntime] = []
    for index, (spec, override) in enumerate(zip(specs, urdf_overrides)):
        urdf_path = override.resolve() if override else _default_urdf(spec.name)
        per_robot_paths = search_roots + _unique_paths([urdf_path.parent, repo_root / "resources" / spec.name])
        initial_state = initial_overrides.get(index)
        runtime = _load_robot(
            client,
            urdf_path,
            spec,
            per_robot_paths,
            args.ignore_visuals,
            initial_configuration=initial_state,
        )
        runtimes.append(runtime)
        if initial_state is not None:
            formatted = ", ".join(f"{value:.4f}" for value in initial_state)
            print(f"Applied initial state to {spec.label}: [{formatted}]")

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    status_id: Optional[int] = None
    last_status: Optional[bool] = None
    env = vamp.Environment()
    _place_obstacles(client, env, obstacle_specs)
    baseline_configs = [runtime.configuration.copy() for runtime in runtimes]
    last_pb_summary: Set[str] = set()
    last_vamp_summary: Set[str] = set()

    if args.randomize:
        _randomize_and_compare(client, runtimes, env, args.randomize, args.random_seed, baseline_configs)
        last_pb_summary.clear()
        last_vamp_summary.clear()
    else:
        try:
            while True:
                states = []
                for runtime in runtimes:
                    for idx, (slider_id, joint_index) in enumerate(zip(runtime.slider_ids, runtime.joint_indices)):
                        value = client.readUserDebugParameter(slider_id)
                        runtime.configuration[idx] = value
                        client.resetJointState(runtime.body_id, joint_index, value)

                    state = {
                        "robot": runtime.spec.name,
                        "configuration": runtime.configuration.tolist(),
                        "transform": runtime.spec.transform.astype(np.float32),
                    }
                    states.append(state)

                try:
                    ts = time.time()
                    collision_free = vamp.fkcc_multi_all(states, env)
                    te = time.time()
                except ValueError as exc:
                    text = f"fkcc_multi_all error: {exc}"
                    color = [1.0, 0.5, 0.0]
                    collision_free = False
                else:
                    text = "Collision Free" if collision_free else "In Collision"
                    color = _status_color(collision_free)

                pb_summary: Optional[Set[str]] = None
                vamp_summary: Optional[Set[str]] = None
                if args.show_collisions:
                    client.performCollisionDetection()
                    pb_summary, pb_lines = _collect_pybullet_contacts(client, runtimes)
                    vamp_summary, vamp_lines = _compute_vamp_contacts(runtimes, env)

                    if pb_summary != last_pb_summary or vamp_summary != last_vamp_summary:
                        if pb_summary:
                            print("PyBullet contacts:")
                            for line in pb_lines:
                                print(f"  {line}")
                        elif last_pb_summary:
                            print("PyBullet contacts: none")

                        if vamp_summary:
                            print("VAMP collisions:")
                            for line in vamp_lines:
                                print(f"  {line}")
                        elif last_vamp_summary:
                            print("VAMP collisions: none")

                        print("---")

                    last_pb_summary = pb_summary or set()
                    last_vamp_summary = vamp_summary or set()

                    text = (
                        f"{text} | VAMP:{len(last_vamp_summary)} PB:{len(last_pb_summary)}"
                    )

                if status_id is not None:
                    client.removeUserDebugItem(status_id)

                anchor = [0.0, 0.0, 0.0]
                parent = -1
                if runtimes:
                    parent = runtimes[0].body_id
                    anchor = [0.0, 0.0, 1.2]

                status_id = client.addUserDebugText(
                    text,
                    textPosition=anchor,
                    textColorRGB=color,
                    textSize=2.0,
                    lifeTime=0,
                    parentObjectUniqueId=parent,
                )

                if args.print and collision_free != last_status:
                    t_cost = (te - ts) * 1000.0
                    print(f"[{time.strftime('%H:%M:%S')}] fkcc_multi_all: {t_cost:.2f} ms")
                    print(f"Collision status changed -> {'free' if collision_free else 'collision'}")
                last_status = collision_free

                client.stepSimulation()
                time.sleep(args.sleep)
        finally:
            client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
