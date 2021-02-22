import warnings

from pathlib import Path

DEFAULT_DATA = Path(__file__).parent.parent / "data"


class Consts:
    _DATA = None

    @property
    def DATA(self):
        if self._DATA:
            return Path(self._DATA).resolve()
        else:
            if DEFAULT_DATA.is_dir():
                warnings.warn(
                    "Default file location is used. Use `sensor_position_dataset_helper.set_data_folder()`to change "
                    "it."
                )
                return DEFAULT_DATA
            raise ValueError("Use `sensor_position_dataset_helper.set_data_folder()` to specify the data location.")
