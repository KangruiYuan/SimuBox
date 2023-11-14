from .agent_base import *


def ABC(pn, pv, input_dict):
    input_dict["Scripts"][pn] = pv

    if pn == "fA":
        fC = fA = pv
        fB = round(1 - fA - fC, 6)
        input_dict["Block"][0]["ContourLength"] = fA
        input_dict["Block"][1]["ContourLength"] = fB
        input_dict["Block"][2]["ContourLength"] = fC

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
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.ABC[
                "C4"
            ]
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                1.0,
                3.78956801,
                3.78956801,
            ]
        elif pv == "Crect":
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.ABC[
                "C4"
            ]
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 4.4, 3.0]
        elif pv == "C2L":
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.ABC[
                "C2L"
            ]
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [1.0, 2.5, 1.8]
        elif pv == "L":
            input_dict["Initializer"]["ModelInitializer"]["Lamellar"] = PhaseInit.ABC[
                "L"
            ]
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

        input_dict["Initializer"]["FileInitializer"] = {
            "Mode": "OMEGA",
            "Path": "phin.txt",
            "SkipLineNumber": SkipLineNumber,
        }

        input_dict["Scripts"]["cal_type"] = "cpu"
        input_dict["Iteration"]["VariableCell"]["VariableCellAcceptance"] = [0.01] * 6
        if pv in ["DG", "CsCl", "NaCl"]:
            input_dict["Initializer"]["Mode"] = "FILE"
            input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [64, 64, 64]
        else:
            input_dict["Initializer"]["Mode"] = "MODEL"
            input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [1, 64, 64]

    elif pn in Cells.__members__.keys():
        input_dict["Initializer"]["UnitCell"]["Length"][Cells[pn].value] = pv
    else:
        print(f"{pn}: invalid.")
