import pytest
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from pandas._testing import assert_frame_equal

from sensor_position_dataset_helper import get_metadata_subject
from sensor_position_dataset_helper.gaitmap_dataset import SensorPositionDatasetMocap, SensorPositionDatasetSegmentation
from .conftest import CACHE, load_or_store_snapshot


@pytest.fixture
def mocap_dataset(dataset_path):
    return SensorPositionDatasetMocap(data_folder=dataset_path, memory=CACHE)


@pytest.fixture
def segmentation_dataset(dataset_path):
    return SensorPositionDatasetSegmentation(data_folder=dataset_path, memory=CACHE)


def test_all_subjects_found(segmentation_dataset):
    assert len(segmentation_dataset) == 14


def test_all_subjects_and_tests_found(mocap_dataset):
    assert len(mocap_dataset) == 14 * 7
    assert len(mocap_dataset.groupby("participant")) == 14


@pytest.mark.parametrize("base_class", (SensorPositionDatasetSegmentation, SensorPositionDatasetMocap))
def test_include_wrong_subject(dataset_path, base_class):
    assert len(base_class(data_folder=dataset_path, include_wrong_recording=True).groupby("participant")) == 15


@pytest.mark.parametrize("base_class", (SensorPositionDatasetSegmentation, SensorPositionDatasetMocap))
def test_sampling_rate(base_class):
    assert base_class().sampling_rate_hz == 204.8


@pytest.mark.parametrize("padding", (0, 2, 5))
def test_test_padding(dataset_path, padding):
    ds = SensorPositionDatasetMocap(data_folder=dataset_path, data_padding_s=padding, memory=CACHE)
    ds = ds.get_subset(participant="4d91", test="fast_10")

    # Actual test length:
    meta_data = get_metadata_subject("4d91", data_folder=dataset_path)
    # Find the start and end index
    test_start = meta_data["imu_tests"]["fast_10"]["start_idx"]
    test_stop = meta_data["imu_tests"]["fast_10"]["stop_idx"]
    actual_length = test_stop - test_start

    expected_length = int(actual_length + 2 * padding * 204.8)
    # As the "real" test extraction is based on the exact timing of the samples, the there might be 1 sample more or
    # less in the actual test
    assert expected_length - 1 <= len(ds.data) <= expected_length + 1


@pytest.mark.parametrize("base_class", (SensorPositionDatasetSegmentation, SensorPositionDatasetMocap))
def test_data_regression(dataset_path, base_class):
    ds = base_class(data_folder=dataset_path, memory=CACHE)
    ds = ds[0]

    # Store first samples of all sensors as regression data
    data = ds.data.iloc[:20]
    for name in get_multi_sensor_names(data.drop("sync", axis=1)):
        compare = load_or_store_snapshot("test_data_regression_{}_{}".format(base_class.__name__, name), data[name])

        assert_frame_equal(data[name], compare)

