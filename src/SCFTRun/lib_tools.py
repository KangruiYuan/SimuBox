import numpy as np
from enum import Enum


def mypara(start, end, step):
    temp_para = np.arange(start, end + step, step)
    while(end < temp_para[-1]):
        temp_para = temp_para[:-1]
    if end not in np.around(temp_para, 6):
        temp_para = np.append(temp_para, end)
    return np.around(temp_para, 6)


class Cells(Enum):
    lx = 0
    ly = 1
    lz = 2
    alpha = 3
    beta = 4
    gamma = 5


class PhaseInit:

    ABC = dict(
        C4=[
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "TopCenter": [0.0, 0.0, 0.0],
                "BottomCenter": [1.0, 0.0, 0.0],
                "Radius": 0.2
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "TopCenter": [0.0, 1.0, 0.0],
                "BottomCenter": [1.0, 1.0, 0.0],
                "Radius": 0.2
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "TopCenter": [0.0, 0.0, 1.0],
                "BottomCenter": [1.0, 0.0, 1.0],
                "Radius": 0.2
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "TopCenter": [0.0, 1.0, 1.0],
                "BottomCenter": [1.0, 1.0, 1.0],
                "Radius": 0.2
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "TopCenter": [0.0, 0.5, 0.5],
                "BottomCenter": [1.0, 0.5, 0.5],
                "Radius": 0.2
            }],
        L=[
            {"ComponentName": "A",
             "Intensity": 0.2,
             "TopCenter": [0.5, 0.5, 0.2],
             "BottomCenter": [0.5, 0.5, 0.35]},
            {"ComponentName": "C",
             "Intensity": 0.2,
             "TopCenter": [0.5, 0.5, 0.65],
             "BottomCenter": [0.5, 0.5, 0.8]}],
        G=[
            {"ComponentName": "A",
             "Intensity": 0.2,
             "Direction": 1.0,
             "Threshold": 0.7,
             "Rescale": [1, 1, 1]},
            {"ComponentName": "C",
             "Intensity": 0.2,
             "Direction": -1.0,
             "Threshold": -0.7,
             "Rescale": [1, 1, 1]}],
        CsCl=[{
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                0.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                1.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                0.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                0.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                1.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                0.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                1.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                1.0,
                1.0
            ]
        },
            {
                "Radius": 0.5,
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    0.5,
                    0.5
                ]
        }],
        NaCl=[
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    0.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    0.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    1.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    0.0,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    1.0,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    0.0,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    1.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    1.0,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    0.5,
                    0.5
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    0.0,
                    0.5
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    0.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.0,
                    0.5,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    1.0,
                    0.5
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    1.0,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    0.5,
                    1.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    1.0,
                    0.5,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    1.0,
                    0.0
                ],
                "Radius": 0.5
            },
            {
                "ComponentName": "C",
                "Intensity": 0.2,
                "Center": [
                    0.5,
                    1.0,
                    0.0
                ],
                "Radius": 0.5
            }
        ]

    )

    AB = dict(
        iHPa=[
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.0, 0.0, 0.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [1.0, 0.0, 0.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.0, 1.0, 0.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [1.0, 1.0, 0.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.5, 0.5, 0.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.0, 0.0, 1.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [1.0, 0.0, 1.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.0, 1.0, 1.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [1.0, 1.0, 1.0]
            },
            {
                "Radius": 0.5,
                "ComponentName": "A",
                "Intensity": 0.2,
                "Center": [0.5, 0.5, 1.0]
            }
        ],
        C6=[
            {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.0, 0.0], "BottomCenter": [
                    1.0, 0.0, 0.0], "Radius": 0.2}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 1.0, 0.0], "BottomCenter": [
                    1.0, 1.0, 0.0], "Radius": 0.2}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.0, 1.0], "BottomCenter": [
                    1.0, 0.0, 1.0], "Radius": 0.2}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 1.0, 1.0], "BottomCenter": [
                    1.0, 1.0, 1.0], "Radius": 0.2}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.5, 0.5], "BottomCenter": [
                    1.0, 0.5, 0.5], "Radius": 0.2}],
        SC=[{
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                0.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                1.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                0.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                0.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                1.0,
                0.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                0.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                0.0,
                1.0,
                1.0
            ]
        }, {
            "Radius": 0.5,
            "ComponentName": "A",
            "Intensity": 0.2,
            "Center": [
                1.0,
                1.0,
                1.0
            ]
        }],
        C3=[
            {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                0.0, 0.16, 0.5], "BottomCenter": [
                1.0, 0.16, 0.5], "Radius": 0.16}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.83, 0.5], "BottomCenter": [
                    1.0, 0.83, 0.5], "Radius": 0.16}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.33, 0.0], "BottomCenter": [
                    1.0, 0.33, 0.0], "Radius": 0.16}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.66, 0.0], "BottomCenter": [
                    1.0, 0.66, 0.0], "Radius": 0.16}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.33, 1.0], "BottomCenter": [
                    1.0, 0.33, 1.0], "Radius": 0.16}, {
                "ComponentName": "A", "Intensity": 0.2, "TopCenter": [
                    0.0, 0.66, 1.0], "BottomCenter": [
                    1.0, 0.66, 1.0], "Radius": 0.16}]
    )


    
