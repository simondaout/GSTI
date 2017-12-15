import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided

from os import path
import sys

from GSTI.date2dec import *
from GSTI.flatten import *

from pyrocko.gf import SatelliteTarget,Target,LocalEngine
from pyrocko import gf, moment_tensor as mtm, trace
from pyrocko import util, pile, model, config, trace, io, pile

logger = logging.getLogger('GSTI.waveforms')


class waveforms:
    def __init__(self, network, reduction, wdir, event,
                 weight=1., phase='P', component='Z',
                 filter_corner=0.055, filter_order=4, filter_type='low',
                 taper_fade=2.0, misfit_norm=2,
                 base=0,
                 sig_base=0,
                 extension='',
                 dist='Unif',
 
                 store_id=None):
        
        self.network=network
        self.reduction=reduction
        self.wdir=wdir
        self.event = event
        self.phase=phase
        self.component=component
        self.filter_corner=filter_corner
        self.filter_order=filter_order
        self.filter_type=filter_type
        self.misfit_norm=misfit_norm
        self.taper_fade=taper_fade
        self.sigmad=1./weight
        
        self.base=base
        self.sig_base=sig_base
        self.Mbase = 1

        self.extension=extension
        self.dist=dist
        self.store_id=store_id

        self.taper = trace.CosFader(xfade=self.taper_fade)  # Cosine taper with fade in and out of 2s.
        self.bw_filter = trace.ButterworthResponse(corner=self.filter_corner,  # in Hz
                                      order=self.filter_order,
                                      type=self.filter_type)  # "low"pass or "high"pass

        self.setup = trace.MisfitSetup(description='Misfit Setup',
                          norm=2,  # L1 or L2 norm
                          taper=self.taper,
                          filter=self.bw_filter,
                          domain='time_domain') # Possible domains are:
                                                # time_domain, cc_max_norm (correlation) 
                                                # and frequency_domain
        self.events = []
        self.events.extend(model.load_events(filename=wdir+self.event))
        origin = gf.Source(
            lat=np.float(self.events[0].lat),
            lon=np.float(self.events[0].lon))
        # print util.time_to_str(events[0].time)

        self.base_source = gf.MTSource.from_pyrocko_event(self.events[0])
        self.base_source.set_origin(origin.lat, origin.lon)
        # print util.time_to_str(self.base_source.time), events[0].lon, events[0].lat
        # sys.exit()
        self.type = 'Waveform'

        self._targets = None

    def get_targets(self):
        if self._targets is None:
            self._targets = []
            for station,tr in zip(stations_list, self.traces):  # iterate over all stations
                # print station.lat, station.lon
                target = Target(
                    lat = np.float(station.lat),  # station lat.
                    lon = np.float(station.lon),   # station lon.
                    store_id = self.store_id,   # The gf-store to be used for this target,
                    # we can also employ different gf-stores for different targets.
                    interpolation = 'multilinear',   # interp. method between gf cells
                    quantity = 'displacement',   # wanted retrieved quantity
                    codes = station.nsl() + ('BH'+self.component,))  # Station and network code

                # Next we extract the expected arrival time for this station from the the store,
                # so we can use this later to define a cut-out window for the optimization:
                self._targets.append(target)
                self.names.append(station.nsl()[1])
        return self._targets

    def load(self,inv):
        # load the data as a pyrocko pile and reform them into an array of traces
        logger.info('Loading waveform data...')
        data = pile.make_pile([self.wdir+self.reduction])
        self.traces = data.all()

        # load station file
        fname = self.wdir + self.network
        stations_list = model.load_stations(fname)

        for s in stations_list:
            s.set_channels_by_name(*self.component.split())

        self.tmin, self.tmax = [], []
        self.arrivals = []
        self.names = []

        # print len(self.traces), len(self.targets)

        self.targets = self.get_targets()

        for station,tr,target in zip(stations_list,self.traces,self.targets):
            
            engine = LocalEngine(store_superdirs=inv.store_path)
            store = engine.get_store(inv.store)
            # trace.snuffle(tr, events=self.events)
            arrival = store.t(self.phase, self.base_source, target)  # expected P-wave arrival
            # print arrival
            tmin = self.base_source.time+arrival-15  # start 15s before theor. arrival
            tmax = self.base_source.time+arrival+15  # end 15s after theor. arrival
            # # print self.tmin,self.tmax
            tr.chop(tmin=tmin, tmax=tmax)
            self.tmin.append(tmin)
            self.tmax.append(tmax)
            self.arrivals.append(self.base_source.time+arrival)

        self.Npoints = len(self.targets)
        # data vector
        self.d = []
        self.d.append(map((lambda x: getattr(x,'ydata')),self.traces))
        self.d=flatten(self.d)
        # time vector
        t = []
        for i in xrange(self.Npoints):
            t.append(self.traces[i].get_xdata())
        # self.t.append(map((lambda x: getattr(x,'get_xdata()')),self.traces))
        # convert time 
        self.t = time2dec(map(util.time_to_str, flatten(t))) 
        # print self.t
        self.N =  len(self.d)
        # print self.t
        # sys.exit()

    def printbase(self):
        print 'Time shift from GCMT:', self.base
        return

    def info(self):
        print
        print 'Waveforms data:',self.network
        print 'Number of stations:', self.Npoints
        print 'Lenght data vector:', self.N
    

    def g(self,inv,m):
        logging.info('Generating G Matrix...')
        m = np.asarray(m)
        # forward vector
        self.gm=np.zeros((self.N))
        # create synth traces for plot and misfit
        self.syn = self.traces
        # set trace to zero
        for tr in self.syn:
            tr.ydata.fill(0.)

        start = 0
        # one seismic trace can be associated to several kernel (ie coseismic+posteseismic)
        for k in xrange(len(inv.kernels)):
            kernel = inv.kernels[k]
            # not sure: slow slips cannot produce seismic waves? so not necessary too loop over it
            # if kernel.seismo is True:
            # one seimic trace can be the result of slip of several patches
            for seg in kernel.segments:
                # update time event
                seg.time += self.base

                synt_traces = inv.process(seg.get_source(), self.targets).pyrocko_traces()
                # for syn,tr in zip(synt_traces,self.traces):
                # print len(syn.ydata), len(tr.ydata)
                # sys.exit()

                temp = 0
                for i in xrange(self.Npoints):
                    # print temp 
                    syn = synt_traces[i]
                    # print len(syn.ydata)
                    # chop synt trace
                    try: 
                        syn.chop(tmin=self.tmin[i], tmax=self.tmax[i])
                        gt = as_strided(self.gm[temp:temp+len(syn.ydata)])
                        time = as_strided(self.t[temp:temp+len(syn.ydata)])
                        gt += syn.ydata*inv.kernels[k].g(time)
                        # print len(gt), gt
                        # print len(self.gm), self.gm
                        # update synt trace: not sure that all synt. trace have same 
                        # reference time...
                        # print util.time_to_str(syn.tmin), util.time_to_str(self.traces[i].tmin)
                        # ref time synt trace = time patch
                        self.syn[i].ydata += syn.ydata*inv.kernels[k].g(time)
                    except:
                        pass
                    
                    temp += len(syn.ydata)

                start += seg.Mpatch

        return self.gm

    def residual(self,inv,m):

        misfit_list = []  # init a list for a all the singular misfits
        norm_list = []  # init a list for a all the singular normalizations
        for tr, syn in zip(self.traces, self.syn):

            misfit, norm = tr.misfit(candidate=syn, setup=self.setup) 
            misfit_list.append(misfit), norm_list.append(norm)  # append the misfit into a list
            
        # self.res = np.asarray(misfit_list)/(self.sigmad*np.asarray(norm_list))
        # print self.res

        g=np.asarray(self.g(inv,m))
        self.res = (self.d-g)/self.sigmad
        # print self.res

        return self.res





