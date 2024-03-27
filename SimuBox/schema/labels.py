from collections import ChainMap

__all__ = [
    "CommonLabels",
    "ABCExtend",
    "AbsLabels",
    "DiffLabels",
    "AbsCommonLabels",
    "DiffCommonLabels",
]

CommonLabels: dict[str, str] = {
    # Feature
    "tau": r"$\tau$",
    "ksi": r"$\xi$",
    "fA": r"$f_{\rm{A}}$",
    "chiN": r"$\chi N$",
    "gamma_B": r"$\gamma _{\rm B}$",
    "phi_AB": r"$\phi _{\rm AB}$",
    "ly": r"$L_x/R_g$",
    "lz": r"$L_y/R_g$",
    # Structure
    "C4": r"${\rm C}^4_{p4mm}$",
    "C6": r"${\rm C}^6_{p6mm}$",
    "Crect": r"${\rm C}^2_{p2mm}$",
    "C3": r"${\rm C}^3_{p3m1}$",
    "iHPa": r"${\rm iHP^a}",
}

ABCExtend = ChainMap(
    {
        "Crect": r"${\rm C}^2_{c2mm}$",
    },
    CommonLabels,
)

AbsLabels = {
    "freeE": r"$F / nk_B T$",
    "freeAB_sum": r"$U / nk_B T$",
    "freeAB": r"$U_{\rm{AB}} / nk_B T$",
    "freeBA": r"$U_{\rm{AB}} / nk_B T$",
    "freeAC": r"$U_{\rm{AC}} / nk_B T$",
    "freeCA": r"$U_{\rm{AC}} / nk_B T$",
    "freeBC": r"$U_{\rm{BC}} / nk_B T$",
    "freeCB": r"$U_{\rm{BC}} / nk_B T$",
    "freeWS": r"$-TS / nk_B T$",
    "bridge": r"$v_B$",
}

DiffLabels = {
    "freeE": r"$\Delta F / nk_B T$",
    "freeAB_sum": r"$\Delta U / nk_B T$",
    "freeAB": r"$\Delta U_{\rm{AB}} / nk_B T$",
    "freeBA": r"$\Delta U_{\rm{AB}} / nk_B T$",
    "freeAC": r"$\Delta U_{\rm{AC}} / nk_B T$",
    "freeCA": r"$\Delta U_{\rm{AC}} / nk_B T$",
    "freeBC": r"$\Delta U_{\rm{BC}} / nk_B T$",
    "freeCB": r"$\Delta U_{\rm{BC}} / nk_B T$",
    "freeWS": r"$-T \Delta S / nk_B T$",
    "bridge": r"$\Delta v_B$",
    "freeAB1": r"$\rm A/B_1$",
    "freeAB2": r"$\rm A/B_2$",
}


AbsCommonLabels = ChainMap(CommonLabels, AbsLabels)
DiffCommonLabels = ChainMap(CommonLabels, DiffLabels)
