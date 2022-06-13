[![PyPI](https://img.shields.io/pypi/v/sensor_position_dataset_helper)](https://pypi.org/project/sensor_position_dataset_helper/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sensor_position_dataset_helper)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# SensorPositionComparison Helper

This is a helper module to extract and handle the data of the [SensorPositionComparison Dataset](https://zenodo.org/record/5747173).

If you use the dataset or this package, please cite:

```
Küderle, Arne, Nils Roth, Jovana Zlatanovic, Markus Zrenner, Bjoern Eskofier, and Felix Kluge.
“The Placement of Foot-Mounted IMU Sensors Does Affect the Accuracy of Spatial Parameters during Regular Walking.”
PLOS ONE 17, no. 6 (June 9, 2022): e0269567. https://doi.org/10.1371/journal.pone.0269567.

```

## Installation and Usage

Install the project via `pip` or `poetry`:

```
pip install sensor_position_dataset_helper
poetry add sensor_position_dataset_helper
```

## Dataset Handling

You also need to download the actual Dataset from [here](https://zenodo.org/record/5747173).
If you are member of the [MaD Lab](https://www.mad.tf.fau.de), you can also get a git-lfs version from 
[our internal server](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison).

Then you need to tell this library about the position of the dataset.
Note that the path should point to the top-level repo folder of the dataset.
This can either be done globally:
```python
from sensor_position_dataset_helper import set_data_folder

set_data_folder("PATH/TO/THE_DATASET")
```

Or on a-per function basis

```python
from sensor_position_dataset_helper import get_all_subjects

get_all_subjects(data_folder="PATH/TO/THE_DATASET")
```

If you are using the tpcp-dataset objects, you need to provide the path in the init.

```python
from sensor_position_dataset_helper.tpcp_dataset import SensorPositionDatasetSegmentation

dataset = SensorPositionDatasetSegmentation(dataset_path="PATH/TO/THE_DATASET")
```

## Code Examples

For simple operations we suggest to use the provided functions.
For a list of functions see the `sensor_position_dataset_helper/helper.py` file.

In this example we load the gait test data of a 4x10 slow gait test of one participant:

```python
from sensor_position_dataset_helper import get_all_subjects, get_metadata_subject, get_imu_test, get_mocap_test

DATA_FOLDER = "PATH/TO/THE_DATASET"
print(list(get_all_subjects(data_folder=DATA_FOLDER)))
# ['4d91', '5047', '5237', '54a9', '6dbe_2', '6e2e', '80b8', '8873', '8d60', '9b4b', 'c9bb', 'cb3d', 'cdfc', 'e54d']

print(list(get_metadata_subject("54a9", data_folder=DATA_FOLDER)["imu_tests"].keys()))
# ['fast_10', 'fast_20', 'long', 'normal_10', 'normal_20', 'slow_10', 'slow_20']

# Finally get the data.
# Note, this thorws a couple of warnings during the data loading, do to the use of custom sensor firmware.
# These warnings can be ignored.
imu_data = get_imu_test("54a9", "slow_10", data_folder=DATA_FOLDER)

mocap_traj = get_mocap_test("54a9", "slow_10", data_folder=DATA_FOLDER)
```

For advanced usage we recommend the use of the `tpcp` datasets.
They provide an object oriented way to access the data and abstract a lot of the complexity that comes with loading the
data.
For general information about object oriented datasets and why they are cool, check out our 
[`tpcp` library](https://github.com/mad-lab-fau/tpcp).

Here we load the same data as above, but using the dataset object:
```python
from sensor_position_dataset_helper.tpcp_dataset import SensorPositionDatasetMocap

DATA_FOLDER = "PATH/TO/THE_DATASET"
ds = SensorPositionDatasetMocap(data_folder=DATA_FOLDER)
print(ds)
# SensorPositionDatasetMocap [98 groups/rows]
#
#      participant       test
#   0         4d91    fast_10
#   1         4d91    fast_20
#   2         4d91       long
#   3         4d91  normal_10
#   4         4d91  normal_20
#   ..         ...        ...
#   93        e54d       long
#   94        e54d  normal_10
#   95        e54d  normal_20
#   96        e54d    slow_10
#   97        e54d    slow_20
#   
#   [98 rows x 2 columns]
#

print(ds.get_subset(participant="54a9"))
# SensorPositionDatasetMocap [7 groups/rows]
#
#     participant       test
#   0        54a9    fast_10
#   1        54a9    fast_20
#   2        54a9       long
#   3        54a9  normal_10
#   4        54a9  normal_20
#   5        54a9    slow_10
#   6        54a9    slow_20
#

data_point = ds.get_subset(participant="54a9", test="slow_10")

# The data is not loaded until here.
# Only when accessing the `.data` or the `marker_position_` attribute the data is loaded.
imu_data = data_point.data
mocap_traj = data_point.marker_position_
```

## Managing Dataset Revisions

To ensure reproducibility, you should save the version of the dataset that was used for a certain analysis.
If you are part of the MaD-Lab and using the internal git-versioned version of the dataset we provide some helpers.

If you are using the version from Zenodo, we unfortunally have no easy way to verify the version and integrity of the
extracted data on disk.
Therefore, make sure to document the version of the Zenodo dataset and verify the md5 hasshum of the zip-file you 
downloaded from Zenodo.

For the git version you can use the helper as follows:

```python
from sensor_position_dataset_helper import ensure_git_revision

ensure_git_revision(data_folder="PATH/TO/THE_DATASET", version="EXPECTED GIT HASH")
```

This will produce an error, if the dataset version you are using is not the one you expect, or if the dataset repo has 
uncommitted changes.
This will prevent bugs, because you accidentally use the wrong dataset version and will directly document the correct 
version.
