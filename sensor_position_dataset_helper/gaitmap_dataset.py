import warnings
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
from gaitmap.future.dataset import Dataset
from gaitmap.utils.datatype_helper import (
    StrideList,
    MultiSensorData,
    PositionList,
    get_multi_sensor_names,
    MultiSensorStrideList,
)
from gaitmap_io.coordinate_system_transformation import COORDINATE_TRANSFORMATION_DICT
from imucal.management import CalibrationWarning
from joblib import Memory
from nilspodlib.exceptions import LegacyWarning, CorruptedPackageWarning, SynchronisationWarning
from scipy.spatial.transform import Rotation

from sensor_position_dataset_helper import (
    get_all_subjects,
    get_session_df,
    get_manual_labels,
    get_all_tests,
    get_imu_test,
    get_manual_labels_for_test,
    get_mocap_test,
    get_subject_mocap_folder,
    get_foot_sensor,
    rotate_dataset,
)


def get_memory(mem):
    if not mem:
        return Memory()
    return mem


def align_coordinates(multi_sensor_data: pd.DataFrame):
    feet = {"r": "right", "l": "left"}
    rotations = {}
    for s in get_multi_sensor_names(multi_sensor_data):
        if "_" not in s:
            continue
        foot, pos = s.split("_")
        rot = COORDINATE_TRANSFORMATION_DICT.get("qualisis_{}_nilspodv1".format(pos), None)
        if not rot:
            continue
        rotations[s] = Rotation.from_matrix(rot["{}_sensor".format(feet[foot])])
    ds = rotate_dataset(multi_sensor_data.drop(columns="sync"), rotations)
    ds["sync"] = multi_sensor_data["sync"]
    return ds


class _SensorPostionDataset(Dataset):
    data_folder: Optional[Union[str, Path]]
    include_wrong_recording: bool
    memory: Optional[Memory]

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        include_wrong_recording: bool = False,
        memory: Optional[Memory] = None,
        groupby: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_folder = data_folder
        self.include_wrong_recording = include_wrong_recording
        self.memory = memory
        super().__init__(groupby=groupby, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def segmented_stride_list_(self) -> MultiSensorStrideList:
        if not self.is_single():
            raise ValueError("Can only get stride lists single participant")
        # TODO: Return this as dictionary
        return self._get_segmented_stride_list(self.index)

    def _get_segmented_stride_list(self, index) -> StrideList:
        raise NotImplementedError()

    @property
    def segmented_stride_list_per_sensor_(self) -> MultiSensorStrideList:
        stride_list = self.segmented_stride_list_
        final_stride_list = {}
        for foot in ["left", "right"]:
            foot_stride_list = stride_list[stride_list["foot"] == foot][["start", "end"]]
            for s in get_foot_sensor(foot):
                final_stride_list[s] = foot_stride_list
        return final_stride_list


class SensorPositionDatasetSegmentation(_SensorPostionDataset):
    @property
    def data(self) -> MultiSensorData:
        if not self.is_single():
            raise ValueError("Can only get data for a single participant")
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
            )
            df = get_memory(self.memory).cache(get_session_df)(
                self.index["participant"].iloc[0], data_folder=self.data_folder
            )
            df = df.reset_index(drop=True)
            df.index /= self.sampling_rate_hz
            df = get_memory(self.memory).cache(align_coordinates)(df)
            return df

    def _get_segmented_stride_list(self, index) -> StrideList:
        stride_list = get_manual_labels(index["participant"].iloc[0], self.data_folder)
        stride_list = stride_list.set_index("s_id")
        return stride_list

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"participant": get_all_subjects(self.include_wrong_recording, data_folder=self.data_folder)}
        )


class SensorPositionDatasetMocap(_SensorPostionDataset):
    data_padding_s: int
    align_data: bool

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        include_wrong_recording: bool = False,
        data_padding_s: int = 0,
        align_data: bool = True,
        memory: Optional[Memory] = None,
        groupby: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_padding_s = data_padding_s
        self.align_data = align_data
        super().__init__(
            data_folder,
            include_wrong_recording=include_wrong_recording,
            memory=memory,
            groupby=groupby,
            subset_index=subset_index,
        )

    @property
    def data(self) -> MultiSensorData:
        """The data per gait test.

        Get the data per gait test.
        If `self.data_padding_s` is set, the extracted data region extends by that amount of second beyond the actual
        gait test.
        Keep that in mind, when aligning data to mocap.
        The time axis is provided in seconds and the 0 will be at the actual start of the gait test.
        """
        if not self.is_single():
            raise ValueError("Can only get data for a single participant")
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
            )
            session_df = get_memory(self.memory).cache(get_session_df)(
                self.index["participant"].iloc[0], data_folder=self.data_folder
            )
            if self.align_data is True:
                session_df = get_memory(self.memory).cache(align_coordinates)(session_df)
            df = get_imu_test(
                self.index["participant"].iloc[0],
                self.index["test"].iloc[0],
                session_df=session_df,
                data_folder=self.data_folder,
                padding_s=self.data_padding_s,
            )
            df = df.reset_index(drop=True)
            df.index /= self.sampling_rate_hz
            df.index -= self.data_padding_s
            df.index.name = "time after start [s]"

            return df

    def _get_segmented_stride_list(self, index) -> StrideList:
        stride_list = get_manual_labels_for_test(
            index["participant"].iloc[0], index["test"].iloc[0], data_folder=self.data_folder
        )
        stride_list = stride_list.set_index("s_id")
        stride_list[["start", "end"]] += int(self.data_padding_s * self.sampling_rate_hz)
        return stride_list

    @property
    def mocap_events_(self) -> StrideList:
        """Get mocap events calculated the Zeni Algorithm.

        Note that the events are provided in mocap samples after the start of the test.
        `self.data_padding_s` is also ignored.
        """
        if not self.is_single():
            raise ValueError("Can only get stride lists single participant")
        mocap_events = get_mocap_events(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.data_folder
        )
        mocap_events = {k: v.drop("foot", axis=1) for k, v in mocap_events.groupby("foot")}
        return mocap_events

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        return 100.0

    @property
    def marker_position_(self) -> PositionList:
        """Get the marker trajectories of a test.

        Note the index is provided in seconds after the start of the test.
        """
        if not self.is_single():
            raise ValueError("Can only get position for lists single participant")
        df = get_memory(self.memory).cache(get_mocap_test)(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.data_folder
        )
        df = df.reset_index(drop=True)
        df.index /= self.mocap_sampling_rate_hz_
        df.index.name = "time after start [s]"
        return df

    def create_index(self) -> pd.DataFrame:
        tests = (
            (p, t)
            for p in get_all_subjects(self.include_wrong_recording, data_folder=self.data_folder)
            for t in get_all_tests(p, self.data_folder)
        )
        return pd.DataFrame(tests, columns=["participant", "test"])
