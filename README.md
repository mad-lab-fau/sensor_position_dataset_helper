# SensorPositionComparison Helper

This is a helper module to extract and handle the data of the
[SensorPositionComparison dataset](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison).

## Installation and Usage

Install the project via `pip` or `poetry:

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/data/SensorPositionComparisonHelper
poetry add git+https://mad-srv.informatik.uni-erlangen.de/MadLab/data/SensorPositionComparisonHelper
```

You also need to download the actual Dataset from [here](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison).

Then you need to tell this library about the position of the dataset:

```python
from sensor_position_dataset_helper import set_data_folder

set_data_folder("PATH/TO/THE_DATASET")
```

You can also overwrite this pass on a per function basis:

```python
from sensor_position_dataset_helper import get_all_subjects

get_all_subjects(data_folder="PATH/TO/THE_DATASET")
```

If you are using the gaitmap-dataset objects, you need to provide the path in the init.

```python
from sensor_position_dataset_helper.gaitmap_dataset import SensorPositionDatasetSegmentation

dataset = SensorPositionDatasetSegmentation(dataset_path="PATH/TO/THE_DATASET")
```