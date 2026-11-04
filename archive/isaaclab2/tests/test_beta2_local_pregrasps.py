from scripts.environments.generate_beta2_local_pregrasps import perturb_local_hand_shape


def test_local_pregrasp_variants_change_relative_geometry_and_are_reproducible():
    hand = {"root_pose": [0.0] * 7, "wrist_target_pose": [0.0] * 7,
            "joint_position": [0.5] * 20, "joint_target": [0.5] * 20}
    source = {"seed": 11, "validated": False, "left": hand, "right": hand,
              "bottle": {"root_pose": [0.0] * 7},
              "metadata": {"asset_variant": "beta2", "geometry_qualified": False,
                           "right_natural_source_seed": 7}}
    one = perturb_local_hand_shape(source, count=4, seed=9101)
    two = perturb_local_hand_shape(source, count=4, seed=9101)
    assert one == two
    assert len({r["seed"] for r in one}) == 4
    assert all(r["bottle"] == source["bottle"] for r in one)
    assert any(r["right"]["joint_position"] != source["right"]["joint_position"] for r in one)
    assert all(r["metadata"]["beta2_local_perturbation"]["same_relative_template_family"]
               for r in one)
