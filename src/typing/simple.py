from enum import Enum


class AppModes(Enum):
    data_plotter = "data plotter"


class Run(str):
    def __new__(cls, *args, **kwargs):
        return super(Run, cls).__new__(cls, *args, **kwargs)


class RunCount(int):
    def __new__(cls, *args, **kwargs):
        return super(RunCount, cls).__new__(cls, *args, **kwargs)