"""Event detection by Zeni et. al for marker based camera systems."""
from typing import Optional

import numpy as np
import pandas as pd

from sensor_position_dataset_helper.internal_helpers import sliding_window_view, enforce_stride_list_consistency


class ZeniEventDetection:
    """Detect relevant events based on marker information.

    The original algorithm is expended by a Zero-Velocity detection.
    Further, we are not using the algorithm to find individual strides, but just events with in the strides.
    Stride candidates must be obtained beforehand.
    It is assumed that in each stride we will encounter the events in the following order: TC, IC, min_vel
    """

    min_vel_search_win_size: int

    segmented_event_list_: pd.DataFrame
    forward_direction_: np.ndarray
    fcc_l5_distance_: np.ndarray
    toe_l5_distance_: np.ndarray

    fcc: np.ndarray
    toe: np.ndarray
    stride_list: pd.DataFrame
    sagittal_plane_normal: Optional[np.ndarray]

    _z_axis: np.ndarray = np.array([0, 0, 1.0])
    _default_sagittal_plane = np.array([0, 1.0, 0])

    def __init__(
        self,
        min_vel_search_win_size: int = 10,
    ):
        self.min_vel_search_win_size = min_vel_search_win_size

    def detect(
        self,
        l5: np.ndarray,
        fcc: np.ndarray,
        toe: np.ndarray,
        stride_list: pd.DataFrame,
        sagittal_plane_normal: Optional[np.ndarray] = None,
    ):
        self.fcc = fcc
        self.toe = toe
        self.stride_list = stride_list
        self.sagittal_plane_normal = sagittal_plane_normal
        if sagittal_plane_normal is None:
            sagittal_plane_normal = self._default_sagittal_plane

        self.forward_direction_ = self._calc_forward(sagittal_plane_normal=sagittal_plane_normal)
        self.fcc_l5_distance_ = self._calc_distance(l5, fcc, self.forward_direction_)
        self.toe_l5_distance_ = self._calc_distance(l5, toe, self.forward_direction_)

        ic = []
        tc = []
        min_vel = []
        stride_list = stride_list.copy()
        for i, stride in stride_list.iterrows():
            start, end = stride[["start", "end"]]
            mid_way = start + (end - start) // 2
            ic.append(self.fcc_l5_distance_[start:end].argmax() + start)
            # Restrict the tc detection to the first half of the stride to avoid false detections
            tc.append(self.toe_l5_distance_[start:mid_way].argmin() + start)
            min_vel.append(self._find_min_vel(self.fcc[start:end], self.toe[start:end]) + start)

        stride_list["ic"] = ic
        stride_list["tc"] = tc
        stride_list["min_vel"] = min_vel

        stride_list, _ = enforce_stride_list_consistency(stride_list, stride_type="segmented")

        self.segmented_event_list_ = stride_list

        return self

    def _calc_forward(self, sagittal_plane_normal):
        return np.cross(sagittal_plane_normal, self._z_axis)

    def _calc_distance(self, l5, marker, forward):
        return np.sum((marker - l5) * forward, axis=1)

    def _find_min_vel(self, fcc, toe):
        min_vel_search_win_size = self.min_vel_search_win_size
        combined = np.hstack([toe, fcc])
        speed = np.diff(combined, axis=0)
        energy = np.sum(speed ** 2, axis=1)
        energy = sliding_window_view(energy, window_length=min_vel_search_win_size, overlap=min_vel_search_win_size - 1)
        # find window with lowest summed energy
        min_vel_start = np.argmin(np.sum(energy, axis=1))
        # min_vel event = middle of this window
        min_vel_center = min_vel_start + min_vel_search_win_size // 2
        return min_vel_center
