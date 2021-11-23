"""This is a collection of helpers that were copied from MaDLab internal codebases.

This specific copy of these code snippets are published under a MIT license.

The original copyright belongs to MaD-DiGait group.
All original authors agreed to have the respective code snippets published in this way.

In case any problems are detected, these changes should be upstreamed to the respective internal libraries.
"""
from typing import Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

#: The default names of the Gyroscope columns in the sensor frame
from typing_extensions import Literal

SF_GYR = ["gyr_x", "gyr_y", "gyr_z"]
#: The default names of the Accelerometer columns in the sensor frame
SF_ACC = ["acc_x", "acc_y", "acc_z"]
#: The default names of all columns in the sensor frame
SF_COLS = [*SF_ACC, *SF_GYR]

#: Expected Order of events based on the stride type
SL_EVENT_ORDER = {
    "segmented": ["tc", "ic", "min_vel"],
    "min_vel": ["pre_ic", "min_vel", "tc", "ic"],
    "ic": ["ic", "min_vel", "tc"],
}


COORDINATE_TRANSFORMATION_DICT = dict(
    qualisis_lateral_nilspodv1={
        # [[+y -> +x], [+z -> +y], [+x -> +z]]
        "left_sensor": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        # [[-y -> +x], [-z -> +y], [+x -> +z]]
        "right_sensor": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    },
    qualisis_medial_nilspodv1={
        # [[-y -> +x], [-z -> +y], [+x -> +z]]
        "left_sensor": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        # [[+y -> +x], [+z -> +y], [+x -> +z]]
        "right_sensor": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    },
    qualisis_instep_nilspodv1={
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "left_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "right_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    },
    qualisis_cavity_nilspodv1={
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "left_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "right_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    },
    qualisis_heel_nilspodv1={
        # [[-z -> +x], [+y -> +y], [+x -> +z]]
        "left_sensor": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        # [[-z -> +x], [+y -> +y], [+x -> +z]]
        "right_sensor": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    },
    qualisis_insole_nilspodv1={
        # [[+y -> +x], [-x -> +y], [+z -> +z]]
        "left_sensor": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        # [[-y -> +x], [+x -> +y], [+z -> +z]]
        "right_sensor": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    },
)


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(dataset: pd.DataFrame, rotation: Union[Rotation, Dict[str, Rotation]]) -> pd.DataFrame:
    """Apply a rotation to acc and gyro data of a dataset.

    Parameters
    ----------
    dataset
        dataframe representing a multiple synchronised sensors.
    rotation
        In case a single rotation object is passed, it will be applied to all sensors of the dataset.
        If a dictionary of rotations is applied, the respective rotations will be matched to the sensors based on the
        dict keys.
        If no rotation is provided for a sensor, it will not be modified.

    Returns
    -------
    rotated dataset
        This will always be a copy. The original dataframe will not be modified.
    """
    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in dataset.columns.unique(level=0)}

    rotated_dataset = dataset.copy()
    original_cols = dataset.columns

    for key in rotation_dict.keys():
        test = _rotate_sensor(dataset[key], rotation_dict[key], inplace=False)
        rotated_dataset[key] = test

    # Restore original order
    rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset


def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
       instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window (must be smaller than array length n)

    overlap : int
        length of desired overlap (must be smaller than window_length)

    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 6.,  7.,  8.,  9., nan]])

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view


def enforce_stride_list_consistency(
    stride_list: pd.DataFrame,
    stride_type=Literal["segmented", "min_vel", "ic"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude those strides where the gait events do not match the expected order or contain NaN.

    Correct order in depends on the stride type:

    - segmented: ["tc", "ic", "min_vel"]
    - min_vel: ["pre_ic", "min_vel", "tc", "ic"]
    - ic: ["ic", "min_vel", "tc"]

    Parameters
    ----------
    stride_list
        A single sensor stride list in a Dataframe format
    stride_type
        Indicate which types of strides are expected to be in the stride list.
        This changes the expected columns and order of events.

    Returns
    -------
    cleaned_stride_list
        stride_list but will all invalid strides removed
    invalid_strides
        all strides that were removed

    """
    order = SL_EVENT_ORDER[stride_type]

    # Note: the following also drops strides that contain NaN for any event
    bool_map = np.logical_and.reduce([stride_list[order[i]] < stride_list[order[i + 1]] for i in range(len(order) - 1)])

    return stride_list[bool_map], stride_list[~bool_map]


def normalize(v: np.ndarray) -> np.ndarray:
    """Simply normalize a vector.

    If a 2D array is provided, each row is considered a vector, which is normalized independently.
    In case an array has norm 0, np.nan is returned.

    Parameters
    ----------
    v : array with shape (3,) or (n, 3)
         vector or array of vectors

    Returns
    -------
    normalized vector or  array of normalized vectors

    Examples
    --------
    1D array

    >>> normalize(np.array([0, 0, 2]))
    array([0., 0., 1.])

    2D array

    >>> normalize(np.array([[2, 0, 0],[2, 0, 0]]))
    array([[1., 0., 0.],
           [1., 0., 0.]])

    0 Array:

    >>> normalize(np.array([0, 0, 0]))
    array([nan, nan, nan])

    """
    v = np.array(v)
    if len(v.shape) == 1:
        ax = 0
    else:
        ax = 1
    # We do not want a warning when we divide by 0 as we expect it
    with np.errstate(divide="ignore", invalid="ignore"):
        return (v.T / norm(v, axis=ax)).T
