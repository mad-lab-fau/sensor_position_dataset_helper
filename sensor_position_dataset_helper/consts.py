from pathlib import Path


class Consts:
    _DATA = None

    @property
    def DATA(self):
        if self._DATA:
            return Path(self._DATA).resolve()
        else:
            raise ValueError("Use `sensor_position_dataset_helper.set_data_folder()` to specify the data location.")
