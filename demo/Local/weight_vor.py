
from src import VoronoiCell

vc = VoronoiCell()
vc.weighted_voronoi_diagrams([[100,350],[100,100],[350,350],[350,100]],
                                        weights=[6000,0,0,0],
                                        plot='vertices',
                                        method='power'
                                        )