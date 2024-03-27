from abc import abstractmethod
from collections import OrderedDict
from typing import Any

from ..Phases import PhaseInit
from ...schema import Cells, Options


class MixinAgent:
    def __call__(self, *args, **kwargs):
        self.modify(*args, **kwargs)

    @abstractmethod
    def modify(self,  pn: str, pv: Any, input_dict: OrderedDict, options: Options):
        pass

class MixinBlockPolymer(MixinAgent):

    def __init__(self, input_dict: OrderedDict):
        self.input_dict = input_dict

    def __getitem__(self, key: str):

        if "chiN" in key:
            return self.input_dict["Component"]["FloryHugginsInteraction"]
        elif "fA" in key:
            return [block for block in self.input_dict["Block"] if block["ComponentName"] == "A"]
        elif "fB" in key:
            return [block for block in self.input_dict["Block"] if block["ComponentName"] == "B"]
        elif "fC" in key:
            return [block for block in self.input_dict["Block"] if block["ComponentName"] == "C"]
        elif key in Cells.__members__.keys():
            return self.input_dict["Initializer"]["UnitCell"]["Length"][Cells[key].value]

    def __setitem__(self, key: str, value):
        if key in Cells.__members__.keys():
            self.input_dict["Initializer"]["UnitCell"]["Length"][Cells[key].value] = value
    def get(self, *args):
        return [self[arg] for arg in args]


def around(number, accuracy: int = 6):
    return round(number, accuracy)

class BABCB_Linear(MixinBlockPolymer):

    def __init__(self, input_dict: OrderedDict):
        super().__init__(input_dict)

    def calculate(self, parameters: dict):
        fC = fA = around(parameters["fA"])
        fB = around(1 - fA - fC)
        fB2 = around(parameters["tau"] * fB)
        fB1 = fB3 = around((fB - fB2)  * 0.5)
        chiN0 = chiN1 = chiN2 = parameters["chiNAC"]
        return {
            "fA": fA,
            "fC": fC,
            "fB1": fB1,
            "fB2": fB2,
            "fB3": fB3,
        }

    def assign(self, parameters: dict):
        ...
