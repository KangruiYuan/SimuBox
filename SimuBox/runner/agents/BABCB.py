from .MixinModel import *

__all__ = ["BABCB"]


class BABCB(MixinAgent):
    def __init__(self, *args, **kwargs):
        self.fB: float = -1.0

    @classmethod
    def modify(cls, pn: str, pv: Any, input_dict: OrderedDict, options: Options):
        if pn == "fA":
            fC = fA = pv
            cls.fB = round(1 - fA - fC, 6)
            input_dict["Block"][1]["ContourLength"] = fA
            input_dict["Block"][3]["ContourLength"] = fC

        elif pn == "tau":
            fB2 = round(pv * cls.fB, 6)
            fB1 = fB3 = round((1 - pv) * cls.fB * 0.5, 6)
            input_dict["Block"][0]["ContourLength"] = fB1
            input_dict["Block"][2]["ContourLength"] = fB2
            input_dict["Block"][4]["ContourLength"] = fB3

        elif pn == "chiNAC":
            input_dict["Component"]["FloryHugginsInteraction"][0][
                "FloryHugginsParameter"
            ] = pv
            input_dict["Component"]["FloryHugginsInteraction"][1][
                "FloryHugginsParameter"
            ] = pv
            input_dict["Component"]["FloryHugginsInteraction"][2][
                "FloryHugginsParameter"
            ] = pv

        elif pn == "phase":
            SkipLineNumber = 1
            if pv == "C4":
                input_dict["Initializer"]["ModelInitializer"][
                    "Cylinder"
                ] = PhaseInit.ABC["C4"]
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 3.0, 3.0]
            elif pv == "Crect":
                input_dict["Initializer"]["ModelInitializer"][
                    "Cylinder"
                ] = PhaseInit.ABC["C4"]
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    1.0,
                    3.510816,
                    2.744868,
                ]
            elif pv == "C2L":
                input_dict["Initializer"]["ModelInitializer"][
                    "Cylinder"
                ] = PhaseInit.ABC["C2L"]
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 2.5, 1.8]
            elif pv == "L":
                input_dict["Initializer"]["ModelInitializer"][
                    "Lamellar"
                ] = PhaseInit.ABC["L"]
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 1.0, 4.3]
            elif pv == "DG":
                # input_dict["Initializer"]["ModelInitializer"]["Gyroid"] = PhaseInit.G_init
                # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Ia_3d"
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    5.163667,
                    5.163667,
                    5.163667,
                ]

            elif pv == "CsCl":
                # input_dict["Initializer"]["ModelInitializer"]["Sphere"] = PhaseInit.CsCl_init
                # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Pm_3n"
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    2.36377,
                    2.36377,
                    2.36377,
                ]

            elif pv == "NaCl":
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    3.822753,
                    3.822753,
                    3.822753,
                ]
                SkipLineNumber = 2

            else:
                print(f"{pv}: invalid.")

            input_dict["Initializer"]["FileInitializer"] = {
                "Mode": "OMEGA",
                "Path": "phin.txt",
                "SkipLineNumber": SkipLineNumber,
            }

            input_dict["Scripts"]["cal_type"] = "cpu"
            input_dict["Iteration"]["VariableCell"]["VariableCellAcceptance"] = [
                0.01
            ] * 6
            if pv in options.init_phin:
                input_dict["Initializer"]["Mode"] = "FILE"
                input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [
                    64,
                    64,
                    64,
                ]
            else:
                input_dict["Initializer"]["Mode"] = "MODEL"
                input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [
                    1,
                    64,
                    64,
                ]

        elif pn in Cells.__members__.keys():
            input_dict["Initializer"]["UnitCell"]["Length"][Cells[pn].value] = pv

        else:
            print(f"{pn}: invalid.")

        if pn not in Cells.__members__.keys():
            input_dict["Scripts"][pn] = pv
