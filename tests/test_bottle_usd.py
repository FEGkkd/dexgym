import importlib.util
from pathlib import Path
import tempfile
import unittest

from dexteroushands_lab.assets.bottle import BottleConfig


@unittest.skipUnless(importlib.util.find_spec("pxr"), "OpenUSD Python bindings unavailable")
class BottleUsdTests(unittest.TestCase):
    def test_d6_joint_locks_four_axes_and_leaves_twist_for_wrench_controller(self):
        from pxr import Usd, UsdPhysics
        from dexteroushands_lab.assets.bottle.generate_usd import author_thread_joint

        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/World/envs/env_0/Bottle", "Xform"))
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/World/envs/env_0/Cap", "Xform"))
        joint = author_thread_joint(stage, "/World/envs/env_0", BottleConfig())
        self.assertEqual(joint.GetBody0Rel().GetTargets()[0].pathString, "/World/envs/env_0/Bottle")
        self.assertEqual(joint.GetBody1Rel().GetTargets()[0].pathString, "/World/envs/env_0/Cap")
        for axis in ("transY", "transZ", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Get(joint.GetPrim(), axis)
            self.assertGreater(limit.GetLowAttr().Get(), limit.GetHighAttr().Get())
        self.assertFalse(joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, "rotX"))

    def test_two_dynamic_rigid_bodies_and_bounded_primitive_colliders(self):
        from pxr import Usd, UsdPhysics
        from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd

        with tempfile.TemporaryDirectory() as directory:
            bottle_path, cap_path = build_bottle_usd(BottleConfig(), Path(directory))
            self.assertTrue(bottle_path.is_file())
            self.assertTrue(cap_path.is_file())
            for path, root_name in ((bottle_path, "Bottle"), (cap_path, "Cap")):
                stage = Usd.Stage.Open(str(path))
                root = stage.GetPrimAtPath("/" + root_name)
                self.assertTrue(root.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertTrue(root.HasAPI(UsdPhysics.MassAPI))
                colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
                self.assertGreater(len(colliders), 10)
                self.assertLess(len(colliders), 60)
                self.assertTrue(all(p.GetTypeName() in ("Capsule", "Cube", "Cylinder") for p in colliders))


if __name__ == "__main__":
    unittest.main()
