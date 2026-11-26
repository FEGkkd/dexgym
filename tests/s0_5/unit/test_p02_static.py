"""Regression checks for P02 semantic identity and analytic gradients."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.intent import (
    CHAINS, preserve_near_pinch, robot_target, semantic_order, target_geometry,
)
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.p02.oracle import Objective


class IntentTest(unittest.TestCase):
    def test_semantic_order_ignores_record_order_and_follows_parent_ids(self):
        nodes=[dict(node_id=99,parent_node_id=99,chain_type='Hand',joint_type='Invalid',original_order=0)]
        for s,chain in enumerate(CHAINS):
            kinds=('MCP','PIP','DIP','TIP') if not s else ('MCP','PIP','IP','DIP','TIP')
            p=99
            for j,k in enumerate(kinds):
                node=100+s*7+j
                nodes.append(dict(node_id=node,parent_node_id=p,chain_type=chain,joint_type=k,original_order=len(nodes)))
                p=node
        root,order=semantic_order({'joints':nodes})
        reordered=semantic_order({'joints':list(reversed(nodes))})
        self.assertEqual(root,reordered[0])
        np.testing.assert_array_equal(order,reordered[1])
        nodes[-1]['parent_node_id']=100
        with self.assertRaises(ValueError):
            semantic_order({'joints':nodes})

    def test_near_pinch_exact_preservation_and_rigid_equivariance(self):
        rng=np.random.default_rng(14)
        robot=rng.normal(size=(5,4,3))*.05
        human=rng.normal(size=(5,4,3))*.05
        human[1,-1]=human[0,-1]+[0.,.004,0.]
        human[2:,-1]+=1.
        rotation=Rotation.from_euler('z',.7).as_matrix()
        corrected,indices,desired=preserve_near_pinch(robot,human,1.3,rotation,.02)
        np.testing.assert_array_equal(indices,[0])
        np.testing.assert_allclose(corrected[1,-1]-corrected[0,-1],desired[0],atol=1e-12)
        np.testing.assert_array_equal(corrected[:,0],robot[:,0])
        np.testing.assert_array_equal(corrected[2:],robot[2:])

    def test_shared_world_preserves_relative_vectors(self):
        w=np.array([[.2,.1,.1],[.5,-.4,.2]])
        r=np.stack((np.eye(3),np.eye(3)))
        hand=np.arange(120).reshape(2,5,4,3)*.0004
        mount=(np.zeros((2,3)),r)
        pairs=[[0,0],[0,2],[2,0]]
        first=target_geometry(w,r,hand,mount,pairs)
        yaw=Rotation.from_euler('z',.34).as_matrix()
        second=target_geometry(w@yaw.T+[0.,-.2,.1],yaw@r,hand,mount,pairs)
        np.testing.assert_allclose(second.cross,first.cross@yaw.T,atol=1e-12)
        np.testing.assert_allclose(second.relative_wrist,first.relative_wrist,atol=1e-12)
        np.testing.assert_allclose(second.pinch,first.pinch,atol=1e-12)


class ActualModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(Path('docs/s0-5/evidence/P02/plan.json').read_bytes())
        cls.robot=RobotModel(cls.config['model'])

    def test_named_actuators_and_coupling(self):
        r=self.robot
        self.assertEqual([len(x) for x in r.groups],[7,7,20,20])
        q=r.home.copy()
        q[5],q[6]=np.deg2rad([58,79])
        self.assertGreater(np.max(r.coupling@q),COUPLING_LIMIT)
        self.assertFalse(r.legality(q)['feasible'])
        self.assertTrue(r.legality(r.home)['feasible'])

    def test_static_objective_matches_independent_directional_derivative(self):
        r=self.robot
        goal=robot_target(r,r.geometry(r.home),self.config['cross_hand_pairs'])
        obj=Objective(r,goal,self.config)
        rng=np.random.default_rng(27)
        q=r.home+rng.normal(size=54)*.035
        d=rng.normal(size=54)
        h=1e-6
        plus=obj.evaluate(q+h*d)[0]
        minus=obj.evaluate(q-h*d)[0]
        analytic=obj.evaluate(q)[1]@d
        np.testing.assert_allclose((plus-minus)/(2*h),analytic,rtol=1e-7,atol=1e-6)
