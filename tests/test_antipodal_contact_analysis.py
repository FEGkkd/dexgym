import math

from dexteroushands_lab.scenes.antipodal_contact_analysis import analyze_bottle_contacts


def _point(x,y,name):
    return {"category":"left_bottle","source_body":"Bottle",
            "partner_body":name,"source_path":"/Bottle","partner_path":"/"+name,
            "point_w_m":[x,y,0.5],"normal_w":[x,y,0],
            "normal_force_n":1.0,"pair_friction_force_vector_w_n":[0,0,.1]}


def test_contact_coverage_rejects_one_side_and_accepts_surrounding_patches():
    pose=(0,0,0.5,0,0,0,1)
    same=[_point(.035,0,"l_index_finger_distal"),
          _point(.03,.02,"l_middle_finger_distal"),
          _point(.03,-.02,"l_ring_finger_distal")]
    a=analyze_bottle_contacts(same,pose)
    assert not a["surrounds_bottle_axis"]
    assert a["maximum_angular_gap_rad"]>math.pi
    surrounding=[_point(.035,0,"l_index_finger_distal"),
                 _point(-.035,0,"l_thumb_distal"),
                 _point(0,.035,"l_middle_finger_distal"),
                 _point(0,-.035,"l_ring_finger_distal")]
    b=analyze_bottle_contacts(surrounding,pose)
    assert b["surrounds_bottle_axis"]
    assert b["radial_force_imbalance_ratio"]<1e-8
    assert b["reported_pair_axial_friction_sum_n"]==.4


def test_frictional_pinch_requires_sidewall_opposition_from_configured_mu():
    pose = (0, 0, 0.5, 0, 0, 0, 1)
    thumb = _point(.035, 0, "l_thumb_distal")
    finger = _point(-.0175, .03031, "l_middle_finger_middle")  # 120 deg
    pinch = analyze_bottle_contacts([thumb, finger], pose,
                                    static_friction=.6, bottle_height=.2)
    assert pinch["frictional_opposition"]
    assert not pinch["surrounds_bottle_axis"]
    assert math.degrees(pinch["frictional_opposition_required_angle_rad"]) > 118
    shallow = analyze_bottle_contacts([thumb, _point(0, .035, "l_middle_finger_middle")],
                                      pose, static_friction=.6, bottle_height=.2)
    assert not shallow["frictional_opposition"]
    rim = {**finger, "point_w_m": [-.0175, .03031, .6]}
    assert not analyze_bottle_contacts([thumb, rim], pose,
                                       static_friction=.6, bottle_height=.2)[
                                           "frictional_opposition"]
