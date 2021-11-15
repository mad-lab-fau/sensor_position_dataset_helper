# SensorPositionComparison Helper

This is a helper module to extract and handle the data of the
[SensorPositionComparison dataset](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison).

## Installation and Usage

Install the project via `pip` or `poetry`:

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/data/SensorPositionComparisonHelper
poetry add git+https://mad-srv.informatik.uni-erlangen.de/MadLab/data/SensorPositionComparisonHelper
```

## Dataset Handling
You also need to download the actual Dataset from [here](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison).

Then you need to tell this library about the position of the dataset.
Note that the path should point to the top-level repo folder of the dataset.

```python
from sensor_position_dataset_helper import set_data_folder

set_data_folder("PATH/TO/THE_DATASET")
```

You can also overwrite this pass on a per function basis:

```python
from sensor_position_dataset_helper import get_all_subjects

get_all_subjects(data_folder="PATH/TO/THE_DATASET")
```

If you are using the tpcp-dataset objects, you need to provide the path in the init.

```python
from sensor_position_dataset_helper.tpcp_dataset import SensorPositionDatasetSegmentation

dataset = SensorPositionDatasetSegmentation(dataset_path="PATH/TO/THE_DATASET")
```

## Managing Dataset Revisions

To ensure reproducibility, you should save the version of the dataset that was used for a certain analysis.
This can be easily done by placing the following line at the top of your script:

```python
from sensor_position_dataset_helper import ensure_git_revision

ensure_git_revision(data_folder="PATH/TO/THE_DATASET", version="EXPECTED GIT HASH")
```

This will produce an error, if the dataset version you are using is not the one you expect, or if the dataset repo has 
uncommitted changes.
This will prevent bugs, because you accidentally use the wrong dataset version and will directly document the correct 
version.

For very important work, it is also a good idea to register the dataset as a git-submodule of your analysis project and
keep it pinned at the correct version that way.