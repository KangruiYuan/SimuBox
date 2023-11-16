from collections import ChainMap

CommonLabels: dict[str, str] = {
    # Feature
    "tau": r"$\tau$",
    "ksi": r"$\xi$",
    "fA": r"$f_{\rm{A}}$",
    "chiN": r"$\chi \rm{N}$",
    "gamma_B": r"$\gamma _{\rm B}$",
    "phi_AB": r"$\phi _{\rm AB}$",
    "ly": r"$L_x/R_g$",
    "lz": r"$L_y/R_g$",

    # Structure
    "C4": r"${\rm C}^4_{p4mm}$",
    "C6": r"${\rm C}^6_{p6mm}$",
    "Crect": r"${\rm C}^2_{p2mm}$",
    "C3": r"${\rm C}^3_{p3m1}$",
}

AbsLabels = {
    "freeE": r"$\rm{F} / \rm{nk_B T}$",
    "freeAB": r"$\rm{U} / \rm{nk_B T}$",
    "freeWS": r"$\rm{-TS} / \rm{nk_B T}$",
    "bridge": r"$v_B$",

}

DiffLabels = {
    "freeE": r"$\Delta \rm{F} / \rm{nk_B T}$",
    "freeAB": r"$\Delta \rm{U} / \rm{nk_B T}$",
    "freeWS": r"$\rm{-T} \Delta \rm{S} / \rm{nk_B T}$",
    "bridge": r"$\Delta v_B$",
    "freeAB1": r"$\rm A/B_1$",
    "freeAB2": r"$\rm A/B_2$",
}

AbsCommon = ChainMap(CommonLabels, AbsLabels)
DiffCommon = ChainMap(CommonLabels, DiffLabels)