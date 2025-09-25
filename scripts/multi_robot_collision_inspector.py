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
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

import vamp
from vamp.transformations import euler_matrix, quaternion_from_matrix


@dataclass
class RobotSpec:
    name: str
    transform: np.ndarray  # 4x4 float32


@dataclass
class RobotRuntime:
    spec: RobotSpec
    module: object
    body_id: int
    joint_names: Sequence[str]
    joint_indices: Sequence[int]
    slider_ids: Sequence[int]
    configuration: np.ndarray  # working buffer


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
    parts = raw.split("@")
    name = parts[0].strip()
    if not name:
        raise ValueError(f"Invalid robot spec '{raw}': missing name")

    translation = np.zeros(3, dtype=np.float32)
    rotation = np.zeros(3, dtype=np.float32)

    if len(parts) >= 2 and parts[1]:
        values = [float(v) for v in parts[1].split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError(f"Robot spec '{raw}' translation must provide 3 comma-separated floats")
        translation = np.asarray(values, dtype=np.float32)

    if len(parts) >= 3 and parts[2]:
        values = [float(v) for v in parts[2].split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError(f"Robot spec '{raw}' rotation must provide 3 comma-separated floats (radians)")
        rotation = np.asarray(values, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = euler_matrix(*rotation)[:3, :3].astype(np.float32)
    transform[:3, 3] = translation

    return RobotSpec(name=name, transform=transform)


def _load_robot(
    client: BulletClient,
    urdf_path: Path,
    spec: RobotSpec,
    search_paths: Sequence[str],
    ignore_visuals: bool,
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
    joint_indices: List[int] = []
    joint_names = getattr(vamp, spec.name).joint_names()
    for joint_index in range(client.getNumJoints(body_id)):
        info = client.getJointInfo(body_id, joint_index)
        joint_map[info[1].decode()] = joint_index

    sliders: List[int] = []
    configuration = np.zeros(len(joint_names), dtype=np.float32)
    for idx, joint_name in enumerate(joint_names):
        bullet_index = joint_map.get(joint_name)
        if bullet_index is None:
            raise RuntimeError(f"Joint '{joint_name}' not found in URDF '{urdf_path}' for robot '{spec.name}'")
        info = client.getJointInfo(body_id, bullet_index)
        lower, upper = info[8], info[9]
        if not (lower < upper) or math.isinf(lower) or math.isinf(upper):
            lower, upper = -math.pi, math.pi
        slider_id = client.addUserDebugParameter(f"{spec.name}[{idx}] {joint_name}", lower, upper, 0.0)
        sliders.append(slider_id)
        joint_indices.append(bullet_index)

    module = getattr(vamp, spec.name)

    return RobotRuntime(
        spec=spec,
        module=module,
        body_id=body_id,
        joint_names=joint_names,
        joint_indices=joint_indices,
        slider_ids=sliders,
        configuration=configuration,
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
    parser.add_argument("--sleep", type=float, default=1.0 / 120.0, help="Loop sleep in seconds (default: 1/120)")
    parser.add_argument("--ignore-visuals", action="store_true", help="Disable URDF visual meshes (avoids missing texture warnings)")
    parser.add_argument("--no-gui", action="store_true", help="Run without the GUI (sliders disabled; not recommended)")
    parser.add_argument("--print", action="store_true", help="Print collision status changes to stdout")
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
    for spec, override in zip(specs, urdf_overrides):
        urdf_path = override.resolve() if override else _default_urdf(spec.name)
        per_robot_paths = search_roots + _unique_paths([urdf_path.parent, repo_root / "resources" / spec.name])
        runtime = _load_robot(client, urdf_path, spec, per_robot_paths, args.ignore_visuals)
        runtimes.append(runtime)

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    status_id: Optional[int] = None
    last_status: Optional[bool] = None
    env = vamp.Environment()

    try:
        while True:
            states = []
            for runtime in runtimes:
                for slider_id, joint_index, idx in zip(runtime.slider_ids, runtime.joint_indices, range(len(runtime.slider_ids))):
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
                collision_free = vamp.fkcc_multi_all(states, env)
            except ValueError as exc:
                text = f"fkcc_multi_all error: {exc}"
                color = [1.0, 0.5, 0.0]
                collision_free = False
            else:
                text = "Collision Free" if collision_free else "In Collision"
                color = _status_color(collision_free)

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
                print(f"Collision status changed -> {'free' if collision_free else 'collision'}")
            last_status = collision_free

            client.stepSimulation()
            time.sleep(args.sleep)
    finally:
        client.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
