
# *SimuBox* for Science Calculation

Github Version: ***0.1.3***

PYPI Version: ***0.1.0***

This kit is designed to facilitate research in my research group. People who need to do similar work don't have to build computing systems over and over again.
There are still many unfinished areas that can be raised, and I will do my best to improve them as soon as possible.

# Guide for Installment

## Python


`pip install SimuBox`

## Requirements

```
python=3.9
pytorch=1.8.2+cu111
pandas >= 1.4
scipy >= 1.10.0
matplotlib >= 3.6.2
numpy >= 1.19.5
opencv-contrib-python >= 4.7.0.72
opencv-python >= 4.5.3.56
pip i >= 3.4.18.6
```

# Usage

## Modules & Functions

- **xml**: tranform *.xml* files generated by galamost into density files
  - XmlTransformer
- **tools**
  - InfoReader
  - Scatter
- **voronoi**
  - VoronoiCell
- **correlation**
  - Corr
- **plotter**
  - *compare*
    - CompareJudger
    - Labels
  - *land*
    - Landscaper
  - *phase*
    - PhaseDiagram
- **SCFTRun**
  - push_job_TOPS
  - repush
  - template
  - lib_tools
  - extract_data

