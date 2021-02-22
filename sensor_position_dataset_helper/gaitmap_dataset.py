import warnings
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
from gaitmap.future.dataset import Dataset
from gaitmap.utils.datatype_helper import StrideList, MultiSensorData, PositionList, get_multi_sensor_names
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


class SensorPositionDatasetSegmentation(Dataset):
    dataset_path: Optional[Union[str, Path]]
    include_wrong_recording: bool
    memory: Optional[Memory]

    def __init__(
        self,
        subset_index: Optional[pd.DataFrame] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        include_wrong_recording: bool = False,
        memory: Optional[Memory] = None,
        select_lvl: Optional[str] = None,
        level_order: Optional[List[str]] = None,
    ):
        self.dataset_path = dataset_path
        self.include_wrong_recording = include_wrong_recording
        self.memory = memory
        super().__init__(subset_index, select_lvl, level_order)

    @property
    def data(self) -> MultiSensorData:
        if not self.is_single():
            raise ValueError("Can only get data for a single participant")
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
            )
            df = get_memory(self.memory).cache(get_session_df)(
                self.index["participant"].iloc[0], data_folder=self.dataset_path
            )
            df = df.reset_index(drop=True)
            df.index /= self.sampling_rate_hz
            df = get_memory(self.memory).cache(align_coordinates)(df)
            return df

    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def segmented_stride_list_(self) -> StrideList:
        if not self.is_single():
            raise ValueError("Can only get stride lists single participant")
        stride_list = get_manual_labels(self.index["participant"].iloc[0], self.dataset_path)
        stride_list = stride_list.set_index("s_id")
        return stride_list

    @property
    def segmented_stride_list_per_sensor_(self) -> StrideList:
        stride_list = self.segmented_stride_list_
        final_stride_list = {}
        for foot in ["left", "right"]:
            foot_stride_list = stride_list[stride_list["foot"] == foot][["start", "end"]]
            for s in get_foot_sensor(foot):
                final_stride_list[s] = foot_stride_list
        return final_stride_list

    def _create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"participant": get_all_subjects(self.include_wrong_recording, data_folder=self.dataset_path)}
        )


class SensorPositionDatasetMocap(Dataset):
    dataset_path: Optional[Union[str, Path]]
    include_wrong_recording: bool
    memory: Optional[Memory]

    def __init__(
        self,
        subset_index: Optional[pd.DataFrame] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        include_wrong_recording: bool = False,
        memory: Optional[Memory] = None,
        select_lvl: Optional[str] = None,
        level_order: Optional[List[str]] = None,
    ):
        self.dataset_path = dataset_path
        self.include_wrong_recording = include_wrong_recording
        self.memory = memory
        super().__init__(subset_index, select_lvl, level_order)

    @property
    def data(self) -> MultiSensorData:
        if not self.is_single():
            raise ValueError("Can only get data for a single participant")
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
            )
            session_df = get_memory(self.memory).cache(get_session_df)(
                self.index["participant"].iloc[0], data_folder=self.dataset_path
            )
            df = get_imu_test(
                self.index["participant"].iloc[0],
                self.index["test"].iloc[0],
                session_df=session_df,
                data_folder=self.dataset_path,
            )
            df = df.reset_index(drop=True)
            df.index /= self.sampling_rate_hz
            df = get_memory(self.memory).cache(align_coordinates)(df)
            return df

    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def segmented_stride_list_(self) -> StrideList:
        if not self.is_single():
            raise ValueError("Can only get stride lists single participant")
        stride_list = get_manual_labels_for_test(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.dataset_path
        )
        stride_list = stride_list.set_index("s_id")
        return stride_list

    @property
    def segmented_stride_list_per_sensor_(self) -> StrideList:
        stride_list = self.segmented_stride_list_
        final_stride_list = {}
        for foot in ["left", "right"]:
            foot_stride_list = stride_list[stride_list["foot"] == foot][["start", "end"]]
            for s in get_foot_sensor(foot):
                final_stride_list[s] = foot_stride_list
        return final_stride_list

    @property
    def mocap_events_(self) -> StrideList:
        if not self.is_single():
            raise ValueError("Can only get stride lists single participant")
        mocap_events = pd.read_csv(
            get_subject_mocap_folder(self.index["participant"].iloc[0]) / (self.index["test"].iloc[0] + "_steps.csv")
        )
        mocap_events = {k: v.drop("foot", axis=1) for k, v in mocap_events.groupby("foot")}
        return mocap_events

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        return 100.0

    @property
    def marker_position_(self) -> PositionList:
        if not self.is_single():
            raise ValueError("Can only get position for lists single participant")
        df = get_memory(self.memory).cache(get_mocap_test)(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.dataset_path
        )
        df = df.reset_index()
        df.index /= self.mocap_sampling_rate_hz_
        return df

    def _create_index(self) -> pd.DataFrame:
        tests = (
            (p, t)
            for p in get_all_subjects(self.include_wrong_recording, data_folder=self.dataset_path)
            for t in get_all_tests(p, self.dataset_path)
        )
        return pd.DataFrame(tests, columns=["participant", "test"])


