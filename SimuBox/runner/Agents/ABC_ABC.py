import math

from .MixinModel import *


class ABC_ABC(MixinAgent):

    chain2_len: float

    @classmethod
    def modify(cls, pn: str, pv: Any, input_dict: OrderedDict, options: Options):
        if pn == "fA1":
            fC1 = fA1 = round(pv, 6)
            fB1 = round(1 - fA1 - fC1, 6)
            for i, j in zip(range(3), [fA1, fB1, fC1]):
                input_dict["Block"][i]["ContourLength"] = j
        elif pn == "gamma":
            cls.chain2_len = round(pv, 6)
        elif pn == "fA2":
            fA2 = fC2 = round(pv, 6)
            fB2 = round(cls.chain2_len - fA2 - fC2, 6)
            for i, j in zip(range(3, 6), [fA2, fB2, fC2]):
                input_dict["Block"][i]["ContourLength"] = j
        elif pn == "z":
            input_dict["Specy"][1]["ChemicalPotential"] = round(float(math.log(pv)), 6)
        elif pn == 'phi0':
            phi0 = round(pv, 6)
            phi1 = round(1 - phi0, 6)
            input_dict['Specy'][0]["VolumeFraction"] = phi0
            input_dict['Specy'][1]["VolumeFraction"] = phi1
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
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    1.0,
                    5.6,
                    5.6,
                ]
            elif pv == "Crect":
                input_dict["Initializer"]["ModelInitializer"][
                    "Cylinder"
                ] = PhaseInit.ABC["C4"]
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 6.0, 5.2]
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
                # input_dict["Initializer"]["ModelInitializer"]["Gyroid"] = PhaseInit.ABC['DG']
                # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Ia_3d"
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    5.163667,
                    5.163667,
                    5.163667,
                ]
            elif pv == "CsCl":
                # input_dict["Initializer"]["ModelInitializer"]["Gyroid"] = PhaseInit.ABC['CsCl']
                # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Pm_3n"
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    2.36377,
                    2.36377,
                    2.36377,
                ]
            elif pv == "NaCl":
                # input_dict["Initializer"]["ModelInitializer"]["Gyroid"] = PhaseInit.ABC['NaCl']
                input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                    3.822753,
                    3.822753,
                    3.822753,
                ]
                SkipLineNumber = 2
            else:
                print(f"{pv}: invalid.")

            input_dict["Solver"]["Ensemble"] = options.ensemble.value

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
                    96,
                    96,
                ]

        elif pn in Cells.__members__.keys():
            input_dict["Initializer"]["UnitCell"]["Length"][Cells[pn].value] = pv
        else:
            print(f"{pn}: invalid.")

        if pn not in Cells.__members__.keys():
            input_dict["Scripts"][pn] = pv
