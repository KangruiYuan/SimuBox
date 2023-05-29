from .lib_tools import Cells
def Mask_AB_A(pn, pv, input_dict):
    if pn == "fA":
        input_dict["Block"][0]['ContourLength'] = pv
        input_dict["Block"][1]['ContourLength'] = round(1-pv, 6)

    elif pn == 'gamma_B':
        input_dict["Block"][2]['ContourLength'] = pv

    elif pn == "xN":
        input_dict["Component"]["FloryHugginsInteraction"][0]["FloryHugginsParameter"] = pv

    elif pn == 'phi_AB':
        input_dict["Specy"][0]["VolumeFraction"] = pv
        input_dict["Specy"][1]["VolumeFraction"] = round(1 - pv, 6)

    elif pn in input_dict["Constraint"].keys():
        input_dict["Constraint"][pn] = pv

    elif pn == "phase":
        input_dict["Initializer"]["Mode"] = "FILE"
        input_dict["Initializer"]["FileInitializer"] = {
            "Mode": "PHI",
            "Path": "phin.txt",
            "SkipLineNumber": 2
        }

        input_dict['_phase_log'] = pv
        input_dict['_which_type'] = 'gpu'

    elif pn in ['No', 'Mark']:
        input_dict[pn] = pv
    else:
        print(f"{pn}: invalid.")


def BABCB(pn, pv, input_dict):
    if pn == "fA":
        fC = fA = pv
        fB = round(1 - fA - fC, 6)
        input_dict["Block"][1]['ContourLength'] = fA
        input_dict["Block"][3]['ContourLength'] = fC

    elif pn == 'tau':
        tau = pv
        fB2 = round(tau * fB, 6)
        fB1 = fB3 = round((1 - tau) * fB * 0.5, 6)
        input_dict["Block"][0]['ContourLength'] = fB1
        input_dict["Block"][2]['ContourLength'] = fB2
        input_dict["Block"][4]['ContourLength'] = fB3
        input_dict['_tau_log'] = tau


    elif pn == "chiNAC":
        chiNAC = pv
        input_dict["Component"]["FloryHugginsInteraction"][0]["FloryHugginsParameter"] = chiNAC
        input_dict["Component"]["FloryHugginsInteraction"][1]["FloryHugginsParameter"] = chiNAC
        input_dict["Component"]["FloryHugginsInteraction"][2]["FloryHugginsParameter"] = chiNAC

    elif pn == "phase":
        SkipLineNumber = 1
        if pv == "C4":
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.C4_init
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                1.0, 3.0, 3.0]
        elif pv == "C6":
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.C6_init
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                1.0, 3.8, 2.2]
        elif pv == 'Crect':
            input_dict["Initializer"]["ModelInitializer"]["Cylinder"] = PhaseInit.C4_init
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                1.0, 3.510816, 2.744868]
        elif pv == 'L':
            input_dict["Initializer"]["ModelInitializer"]["Lamellar"] = PhaseInit.L_init
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                1.0, 1.0, 4.3]
        elif pv == 'DG':
            # input_dict["Initializer"]["Mode"] = "MODEL"
            # input_dict["Initializer"]["ModelInitializer"]["Gyroid"] = PhaseInit.G_init
            # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Ia_3d"
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                5.163667,
                5.163667,
                5.163667]

        elif pv == 'CsCl':
            # input_dict["Initializer"]["Mode"] = "MODEL"
            # input_dict["Initializer"]["ModelInitializer"]["Sphere"] = PhaseInit.CsCl_init
            # input_dict["Solver"]["PseudospectralMethod"]["AcceptedSymmetry"] = "Cubic_Pm_3n"
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                2.36377, 2.36377, 2.36377]

        elif pv == 'NaCl':
            input_dict["Initializer"]["UnitCell"]["Length"][0:3] = [
                3.822753, 3.822753, 3.822753]
            SkipLineNumber = 2

        else:
            print(f"{pv}: invalid.")

        input_dict["Initializer"]["FileInitializer"] = {
            "Mode": "OMEGA",
            "Path": pv + "_phin.txt",
            "SkipLineNumber": SkipLineNumber
        }

        input_dict['_phase_log'] = pv
        input_dict['_which_type'] = 'cpu'
        input_dict["Iteration"]["VariableCell"]["VariableCellAcceptance"] = [
                                                                                0.01] * 6
        if pv in ['DG', 'CsCl', 'NaCl']:
            input_dict["Iteration"]["VariableCell"]["VariableCellAcceptance"] = [
                                                                                    0.05] * 6
            input_dict["Initializer"]["Mode"] = "FILE"
            input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [
                64, 64, 64]
        else:
            input_dict["Initializer"]["Mode"] = "MODEL"
            input_dict["Solver"]["PseudospectralMethod"]["SpaceGridSize"] = [
                1, 64, 64]

    elif pn in Cells.__members__.keys():
        input_dict["Initializer"]["UnitCell"]["Length"][Cells[pn].value] = pv

    else:
        print(f"{pn}: invalid.")