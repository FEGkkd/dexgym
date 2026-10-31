"""Optional live GUI markers for fingertip and bottle contact diagnostics."""

from __future__ import annotations

import math

from dexteroushands_lab.scenes.fingertip_sites import _rotate


def _quat_z_to(direction):
    """Return xyzw rotation taking +Z to a nonzero world direction."""
    length = math.sqrt(sum(x*x for x in direction))
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z = (v/length for v in direction)
    if z < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sqrt(2*(1+z))
    return (-y/scale, x/scale, 0.0, scale/2)


class GraspDebugMarkers:
    """Show five live tips, target surface, bottle axis and contact normals."""

    def __init__(self, prim_path="/Visuals/GraspDiagnostics"):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg

        sphere = lambda radius, rgb: sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))
        self.points = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path=prim_path + "/Points",
            markers={"tip_free": sphere(.003, (0.0, 0.8, 0.2)),
                     "tip_close": sphere(.003, (0.1, 0.3, 1.0)),
                     "tip_inside": sphere(.003, (1.0, 0.2, 0.1)),
                     "surface": sphere(.0015, (0.0, 0.8, 1.0)),
                     "contact": sphere(.0025, (1.0, 0.2, 0.0))},
        ))
        self.lines = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path=prim_path + "/Axes",
            markers={"axis": sim_utils.CylinderCfg(
                         radius=.0008, height=1.0,
                         visual_material=sim_utils.PreviewSurfaceCfg(
                             diffuse_color=(1.0, 0.8, 0.0))),
                     "normal": sim_utils.CylinderCfg(
                         radius=.0006, height=1.0,
                         visual_material=sim_utils.PreviewSurfaceCfg(
                             diffuse_color=(1.0, 0.1, 0.7)))},
        ))

    def update(self, tips, bottle_pose, bottle_radius, bottle_height, contacts):
        import numpy as np

        center = bottle_pose[:3]
        axis = _rotate(bottle_pose[3:7], (0.0, 0.0, 1.0))
        points, indices = [], []
        for tip in tips.values():
            point = tip["site_position_w_m"]
            distance = tip["signed_distance_to_cylindrical_surface_m"]
            points.append(point)
            indices.append(2 if distance < -.002 else 1 if distance < .004 else 0)
            axial = tip["axial_position_bottle_local_m"]
            on_axis = tuple(center[i] + axial*axis[i] for i in range(3))
            radial = tuple(point[i] - on_axis[i] for i in range(3))
            r = math.sqrt(sum(x*x for x in radial))
            if r > 1e-8:
                points.append(tuple(on_axis[i] + bottle_radius*radial[i]/r
                                    for i in range(3)))
                indices.append(3)
        for contact in contacts:
            points.append(contact["point_w_m"])
            indices.append(4)
        self.points.visualize(translations=np.asarray(points, dtype=np.float32),
                              marker_indices=indices)
        line_points = [tuple(center[i] + axis[i]*bottle_height/2 for i in range(3))]
        line_quats = [_quat_z_to(axis)]
        line_scales = [(1.0, 1.0, bottle_height)]
        line_indices = [0]
        for contact in contacts:
            normal = contact["normal_w"]
            point = contact["point_w_m"]
            line_points.append(tuple(point[i] + .0075*normal[i] for i in range(3)))
            line_quats.append(_quat_z_to(normal))
            line_scales.append((1.0, 1.0, .015))
            line_indices.append(1)
        self.lines.visualize(translations=np.asarray(line_points, dtype=np.float32),
                             orientations=np.asarray(line_quats, dtype=np.float32),
                             scales=np.asarray(line_scales, dtype=np.float32),
                             marker_indices=line_indices)
