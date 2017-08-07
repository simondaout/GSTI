from pyrocko import gf
from pyrocko.guts import List
import numpy as num

class CombiSource(gf.Source):
    discretized_source_class = gf.DiscretizedMTSource

    subsources = List.T(gf.Source.T())

    def __init__(self, subsources=[], **kwargs):
        if subsources:

            lats = num.array([subsource.lat for subsource in subsources], dtype=num.float)
            lons = num.array([subsource.lon for subsource in subsources], dtype=num.float)

            if num.all(lats == lats[0]) and num.all(lons == lons[0]):
                lat, lon = lats[0], lons[0]
            else:
                lat, lon = center_latlon(subsources)

            depth = float(num.mean([p.depth for p in subsources]))
            t = float(num.mean([p.time for p in subsources]))
            kwargs.update(time=t, lat=float(lat), lon=float(lon), depth=depth)

        gf.Source.__init__(self, subsources=subsources, **kwargs)

    def get_factor(self):
        return 1.0

    def discretize_basesource(self, store, target=None):

        dsources = []
        t0 = self.subsources[0].time
        for sf in self.subsources:
            assert t0 == sf.time
            ds = sf.discretize_basesource(store, target)
            ds.m6s *= sf.get_factor()
            dsources.append(ds)

        return gf.DiscretizedMTSource.combine(dsources)