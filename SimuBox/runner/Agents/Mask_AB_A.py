from .MixinModel import *
def Mask_AB_A(pn, pv, input_dict):
    input_dict["Scripts"][pn] = pv

    if pn == "fA":
        input_dict["Block"][0]["ContourLength"] = pv
        input_dict["Block"][1]["ContourLength"] = round(1 - pv, 6)

    elif pn == "gamma_B":
        input_dict["Block"][2]["ContourLength"] = pv

    elif pn == "xN":
        input_dict["Component"]["FloryHugginsInteraction"][0][
            "FloryHugginsParameter"
        ] = pv

    elif pn == "phi_AB":
        input_dict["Specy"][0]["VolumeFraction"] = pv
        input_dict["Specy"][1]["VolumeFraction"] = round(1 - pv, 6)

    elif pn in input_dict["Constraint"].keys():
        input_dict["Constraint"][pn] = pv

    elif pn == "phase":
        input_dict["Initializer"]["Mode"] = "FILE"
        input_dict["Initializer"]["FileInitializer"] = {
            "Mode": "PHI",
            "Path": "phin.txt",
            "SkipLineNumber": 2,
        }
        input_dict["Scripts"]["cal_type"] = "gpu"

    elif pn in ["No", "Mark"]:
        input_dict[pn] = pv
    else:
        print(f"{pn}: invalid.")