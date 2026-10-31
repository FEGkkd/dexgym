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
