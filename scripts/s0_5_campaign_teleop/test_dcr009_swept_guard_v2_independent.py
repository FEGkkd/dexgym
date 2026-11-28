#!/usr/bin/env python3
"""Independent offline controls for the exact V2 proof and unchanged native gate."""

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic_ns
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from check_dcr009_frames import LIVE_SHA, SIGNED_SHA
from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT, digest
from swept_live_guard import EXACT_MODEL_RECEIPT_SHA, Guard, Reason, verified_robot
from swept_live_guard_v2 import GuardV2, SweptEnvelopeV2
from test_swept_live_guard import CROSSING_END, CROSSING_START, packet


CLEARANCE_M = 0.0005
ZERO = np.zeros(54)
HERE = Path(__file__).resolve().parent


def state(angle):
    q = ZERO.copy()
    q[0] = angle
    return q


def deadline():
    return monotonic_ns() + 500_000_000


def geometry_fixture(angle=0.0, radial_offset=0.0, common_ancestor=False):
    """Two named spheres with measured 1D gap along a 54D state contract."""
    x, y = (0.1 + radial_offset) * np.array([np.cos(angle), np.sin(angle)])
    target = f'<geom name="target" type="sphere" size=".01" pos="{x} {y} 0"/>'
    moving = '''<body name="moving"><joint name="joint_0" axis="0 0 1"/>
                <geom name="swing" type="sphere" size=".01" pos=".1 0 0"/>'''
    first_axis = moving + (target if common_ancestor else '') + '</body>'
    if not common_ancestor:
        first_axis += target
    other_axes = ''.join(
        f'<body pos="0 0 {i}"><joint name="joint_{i}"/>'
        '<geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>'
        for i in range(1, 54)
    )
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{first_axis}{other_axes}</worldbody></mujoco>')
    first, second = m.geom('swing').id, m.geom('target').id
    return SimpleNamespace(
        model=m, joints=[m.joint(f'joint_{i}') for i in range(54)],
        q_indices=np.arange(54), geom_a=np.array([first]), geom_b=np.array([second]),
        radius=np.array([m.geom_rbound[first] + m.geom_rbound[second]]),
        clearance_m=CLEARANCE_M, has_plane=np.array([False]),
    )


def signed_gap(robot, q):
    data = mujoco.MjData(robot.model)
    data.qpos[robot.q_indices] = q
    mujoco.mj_kinematics(robot.model, data)
    return mujoco.mj_geomDistance(robot.model, data, int(robot.geom_a[0]), int(robot.geom_b[0]), 1.0, None)


def skew_hinge_fixture():
    other_axes = ''.join(
        f'<body pos="0 0 {i}"><joint name="joint_{i}"/>'
        '<geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>'
        for i in range(1, 54)
    )
    model = mujoco.MjModel.from_xml_string(f'''
        <mujoco><worldbody>
        <geom name="target" type="sphere" size=".01" pos=".35 .3 .45"/>
        <body pos=".12 -.08 .15" quat=".8660254 .5 0 0">
          <joint name="joint_0" axis=".6 0 .8" pos=".02 .03 -.07"/>
          <body pos="0 .13 .15" quat=".92388 0 .382683 0">
            <geom name="swing" type="box" size=".01 .02 .03" pos=".16 .031 .40"/>
          </body>
        </body>
        {other_axes}
        </worldbody></mujoco>''')
    first, second = model.geom('swing').id, model.geom('target').id
    return SimpleNamespace(
        model=model, joints=[model.joint(f'joint_{i}') for i in range(54)], q_indices=np.arange(54),
        geom_a=np.array([first]), geom_b=np.array([second]),
        radius=np.array([model.geom_rbound[first] + model.geom_rbound[second]]),
        clearance_m=CLEARANCE_M, has_plane=np.array([False]),
    )


class GeometryControls(unittest.TestCase):
    def test_penetration_inside_interval_with_clear_endpoints_and_midpoint(self):
        for angle in (-0.41, 0.41):
            with self.subTest(angle=angle):
                robot = geometry_fixture(angle)
                guard = SweptEnvelopeV2(robot)
                low, high = state(-0.8), state(0.8)
                safe = state(-np.sign(angle) * 0.6)
                for q in (low, high, ZERO, safe):
                    self.assertGreater(signed_gap(robot, q), CLEARANCE_M)
                self.assertLess(signed_gap(robot, state(angle)), 0.0)
                self.assertEqual(guard.prove(safe, safe, deadline()), Reason.PERMIT)
                # Exercise both children and a cache warmed on the clear side.
                self.assertEqual(guard.prove(low, high, deadline()), Reason.COLLISION)
                self.assertEqual(guard.last_metrics['sampled_clearance_violation'], 'PENETRATION')
                self.assertEqual(guard.prove(safe, safe, deadline()), Reason.PERMIT)

    def test_positive_clearance_shortfall_is_rejected(self):
        robot = geometry_fixture(radial_offset=0.02025)
        exact = signed_gap(robot, ZERO)
        self.assertAlmostEqual(exact, 0.00025, places=10)
        guard = SweptEnvelopeV2(robot)
        self.assertEqual(guard.prove(ZERO, ZERO, deadline()), Reason.COLLISION)
        self.assertEqual(guard.last_metrics['sampled_clearance_violation'], 'CLEARANCE_VIOLATION')

    def test_incomplete_search_fails_closed_at_depth_and_query_caps(self):
        robot = geometry_fixture(angle=0.41)
        for configuration, termination in (
            ({'max_depth': 0}, 'MAX_SUBDIVISION_DEPTH'),
            ({'max_queries': 0}, 'MAX_SUBDIVISION_QUERIES'),
        ):
            with self.subTest(termination=termination):
                guard = SweptEnvelopeV2(robot, **configuration)
                self.assertEqual(guard.prove(state(-0.8), state(0.8), deadline()), Reason.SWEEP_UNPROVEN)
                self.assertEqual(guard.last_metrics['termination'], termination)

    def test_margin_is_retained_on_both_sides(self):
        for gap, expected in ((0.000501, Reason.SWEEP_UNPROVEN), (0.00051, Reason.PERMIT)):
            with self.subTest(gap=gap):
                robot = geometry_fixture(radial_offset=0.02 + gap)
                self.assertAlmostEqual(signed_gap(robot, ZERO), gap, places=10)
                self.assertEqual(SweptEnvelopeV2(robot).prove(ZERO, ZERO, deadline()), expected)

    def test_shared_hinge_cancels_and_expiry_remains_fail_closed(self):
        robot = geometry_fixture(radial_offset=0.0206, common_ancestor=True)
        guard = SweptEnvelopeV2(robot)
        low, high = state(-2.5), state(2.5)
        # The pair stays at .6 mm under the shared transform; large irrelevant axes also vary.
        low[1:], high[1:] = -2, 2
        for q in (low, ZERO, high):
            self.assertAlmostEqual(signed_gap(robot, q), 0.0006, places=10)
        self.assertEqual(guard.prove(low, high, deadline()), Reason.PERMIT)
        self.assertEqual(guard.prove(low, high, monotonic_ns() - 1), Reason.TIME_BUDGET)

    def test_tightened_skew_hinge_bounds_box_corner_displacement(self):
        robot = skew_hinge_fixture()
        guard = SweptEnvelopeV2(robot)
        first = int(robot.geom_a[0])
        corners = np.array(list(product((-1.0, 1.0), repeat=3))) * robot.model.geom_size[first]
        data = mujoco.MjData(robot.model)

        def world_corners(position):
            data.qpos[:] = state(position)
            mujoco.mj_kinematics(robot.model, data)
            return corners @ data.geom_xmat[first].reshape(3, 3).T + data.geom_xpos[first]

        self.assertLess(guard.weights[0, 0], guard.native.weights[0, 0] - 0.05)
        rng = np.random.default_rng(709009)
        for angle, step in rng.uniform(-3.5, 3.5, size=(128, 2)):
            observed = np.linalg.norm(world_corners(angle) - world_corners(angle + step), axis=1).max()
            self.assertLessEqual(observed, guard._variation(state(abs(step)))[0])


class DualModelControls(unittest.TestCase):
    robots = None

    def setUp(self):
        self.home = self.robots[0].home.copy()

    def test_group_sequences_hard_limits_coupling_and_expiry(self):
        for guard_type in (Guard, GuardV2):
            with self.subTest(guard=guard_type.__name__):
                guard = guard_type(self.robots)
                home = self.home
                accepted = guard.evaluate(packet(1, 0, home, home, home), deadline())
                self.assertEqual(accepted[1], Reason.PERMIT)
                guard.publish_decision(accepted[1], accepted[3])
                invalid = home.copy()
                invalid[0] = min(robot.bounds[0, 1] for robot in self.robots) + 0.001
                self.assertEqual(guard.evaluate(packet(2, 0, home, home, invalid), deadline())[1], Reason.LIMIT_OR_COUPLING)
                invalid[0] = np.nan
                self.assertEqual(guard.evaluate(packet(3, 0, home, home, invalid), deadline())[1], Reason.NONFINITE)
                altered = home.copy()
                altered[14] += 0.001
                self.assertEqual(guard.evaluate(packet(4, 0, home, home, altered), deadline())[1], Reason.GROUP_MISMATCH)
                self.assertEqual(guard.evaluate(packet(4, 1, home, home, altered), deadline())[1], Reason.SEQUENCE)
                self.assertEqual(guard.evaluate(packet(5, 0, altered, altered, altered), deadline())[1], Reason.START_UNBOUND)
                self.assertEqual(guard.evaluate(packet(6, 0, home, home, home), monotonic_ns() - 1)[1], Reason.TIME_BUDGET)
                # Each member satisfies hard limits and the coupled rule on its own, but
                # a cross-corner of the asynchronous q box violates J6/J7 coupling.
                first, second = home.copy(), home.copy()
                first[5:7] = [0.98, 0.0]
                second[5:7] = [0.0, 0.98]
                for robot in self.robots:
                    self.assertTrue(np.all(first >= robot.bounds[:, 0]) and np.all(first <= robot.bounds[:, 1]))
                    self.assertTrue(np.all(second >= robot.bounds[:, 0]) and np.all(second <= robot.bounds[:, 1]))
                    self.assertLess(float(max((robot.coupling @ first).max(), (robot.coupling @ second).max())), COUPLING_LIMIT)
                guard.issued_baselines = (first.copy(),)
                self.assertEqual(guard.evaluate(packet(7, 0, first, first, second), deadline())[1], Reason.LIMIT_OR_COUPLING)

    def test_real_asset_interior_collision(self):
        start, end = self.home.copy(), self.home.copy()
        start[:14], end[:14] = CROSSING_START, CROSSING_END
        for guard_type in (Guard, GuardV2):
            with self.subTest(guard=guard_type.__name__):
                guard = guard_type(self.robots)
                for envelope in guard.envelopes:
                    proof = envelope.native
                    self.assertGreater(proof.pose_gap(start), CLEARANCE_M)
                    self.assertGreater(proof.pose_gap(end), CLEARANCE_M)
                    self.assertLess(proof.pose_gap((start + end) * 0.5), 0.0)
                self.assertEqual(guard.evaluate(packet(1, 0, start, start, end), deadline())[1], Reason.COLLISION)

    def test_signed_model_can_veto_an_otherwise_permitted_packet(self):
        guard = GuardV2(self.robots)
        home = self.home
        live, signed = guard.envelopes
        # Exercise the signed-model denial path independently of coincident model verdicts.
        signed.prove = lambda *_: Reason.SWEEP_UNPROVEN
        returned = guard.evaluate(packet(1, 0, home, home, home), deadline())
        self.assertEqual(live.last_metrics['termination'], 'PERMIT')
        self.assertEqual(returned[1], Reason.SWEEP_UNPROVEN)
        guard.publish_decision(returned[1], returned[3])
        self.assertEqual(guard.issued_baselines, ())


NATIVE_FIXTURE = r'''
#include "dcr009_teleop_guard.hpp"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using Guard = dcr009_diagnostic::NominalGuard;
using Names = std::vector<std::string>;
using Positions = std::vector<double>;
using namespace std::chrono_literals;

struct Sources {
  std::array<Names, 4> names;
  Sources() {
    for (const auto * prefix : {"left_j", "right_j"}) {
      const auto group = prefix[0] == 'l' ? 0 : 1;
      for (int i = 1; i <= 7; ++i) names[group].push_back(prefix + std::to_string(i));
    }
    for (const auto * prefix : {"l_", "r_"}) {
      const auto group = prefix[0] == 'l' ? 2 : 3;
      for (const auto * digit : {"THJ", "FFJ", "MFJ", "RFJ", "LFJ"}) {
        for (int i = 0; i <= 3; ++i) names[group].push_back(std::string(prefix) + digit + std::to_string(i));
      }
    }
  }
  void sample(Guard & guard, size_t group, int64_t stamp, int64_t ros_now, bool valid = true) const {
    guard.observe(group, stamp, ros_now, names[group], Positions(names[group].size()), valid);
  }
  void complete(Guard & guard, int64_t stamp, int64_t ros_now) const {
    for (size_t group = 0; group < names.size(); ++group) sample(guard, group, stamp, ros_now);
  }
  Guard::Proof propose(Guard & guard, int64_t ros_now) const {
    Names arm = names[0];
    arm.insert(arm.end(), names[1].begin(), names[1].end());
    return guard.propose(1, "s05-dcr009-independent-v2", 0, ros_now, arm, Positions(14));
  }
};

void expect(const char * name, Guard::Proof proof, uint32_t reason) {
  std::cout << name << ' ' << proof.reason << '\n';
  if (proof.permitted || proof.reason != reason) std::exit(1);
}

int main() {
  const Sources s;
  constexpr int64_t t = 2'000'000'000;
  Guard mixed;
  for (size_t group = 0; group < 4; ++group) s.sample(mixed, group, t - group, t);
  expect("mixed-headers", s.propose(mixed, t), Guard::kNoProof);
  Guard invalid;
  s.complete(invalid, t, t);
  s.sample(invalid, 2, t, t);
  expect("duplicate-group", s.propose(invalid, t), Guard::kNoProof);
  Guard bad;
  for (size_t group = 0; group < 4; ++group) s.sample(bad, group, t, t, group != 2);
  expect("invalid-group", s.propose(bad, t), Guard::kNoProof);
  Guard future;
  s.complete(future, t + 1, t);
  expect("future-feedback", s.propose(future, t), Guard::kNoProof);
  Guard stale;
  s.complete(stale, t, t);
  expect("stale-source-clock", s.propose(stale, t + Guard::kFreshNs + 1), Guard::kNoProof);
  Guard chronological;
  s.complete(chronological, t, t);
  // Freshness passed because only the deliberately absent private transport rejects.
  expect("fresh-complete-reaches-transport", s.propose(chronological, t + 1), Guard::kTransport);
  expect("clock-reversal", s.propose(chronological, t), Guard::kNoProof);
  Guard aged;
  s.complete(aged, t, t);
  std::this_thread::sleep_for(50ms);
  expect("stale-steady-clock", s.propose(aged, t), Guard::kNoProof);
}
'''


def native_controls(compiler):
    from os import environ

    with TemporaryDirectory(prefix='s05-v2-native-') as temporary:
        root = Path(temporary)
        source = root / 'checks.cpp'
        source.write_text(NATIVE_FIXTURE, encoding='utf8')
        output = root / 'native-control'
        subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread', '-I', str(HERE), str(source), '-o', str(output)], check=True)
        environment = dict(environ)
        for key in ('ROS_STATIC_PEERS', 'ROS_DISCOVERY_SERVER', 'CYCLONEDDS_URI', 'FASTRTPS_DEFAULT_PROFILES_FILE',
                    'FASTDDS_DEFAULT_PROFILES_FILE', 'S05_DCR009_SEGMENTED_START_STATE'):
            environment.pop(key, None)
        environment.update(S05_DCR009_GUARD_SOCKET=str(root / 'absent-proof.sock'), S05_DCR009_GUARD_TIMEOUT_MS='30',
                           ROS_DOMAIN_ID='206', ROS_LOCALHOST_ONLY='1', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
        subprocess.run([str(output)], env=environment, check=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-model', required=True, type=Path)
    parser.add_argument('--signed-model', required=True, type=Path)
    parser.add_argument('--model-receipt', required=True, type=Path)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    if digest(args.model_receipt) != EXACT_MODEL_RECEIPT_SHA:
        parser.error('Unexpected exact-model receipt hash')
    expected = json.loads(args.model_receipt.read_bytes())['model_identity']
    DualModelControls.robots = (
        verified_robot(args.live_model, LIVE_SHA, expected['live']),
        verified_robot(args.signed_model, SIGNED_SHA, expected['signed']),
    )
    tests = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    native_controls(args.cxx)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
