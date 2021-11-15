import numpy as np

from sensor_position_dataset_helper.internal_helpers import normalize


def find_plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Get the normal vector of a plane defined by three points."""
    v1 = p2 - p1
    v2 = p3 - p1

    return normalize(np.cross(v1, v2))


def find_sagittal_plane(l5_marker_pos, r_ias_marker_pos, l_ias_marker_pos):
    # TODO: Test
    transversal = find_plane_from_points(l5_marker_pos, r_ias_marker_pos, l_ias_marker_pos)
    midpoint = r_ias_marker_pos - 0.5 * (r_ias_marker_pos - l_ias_marker_pos)
    return normalize(np.cross(transversal, l5_marker_pos - midpoint, axis=1))
