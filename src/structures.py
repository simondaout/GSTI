import numpy as np
import math,sys
import pymc
import operator

from GSTI.flatten import *

from pyrocko import gf, util
from pyrocko.guts import List
from pyrocko.gf import LocalEngine, StaticTarget, SatelliteTarget,\
        RectangularSource

km = 1000.

class patch:
    def __init__(self,name,ss,ds,east,north,down,length,width,strike,dip,
        sig_ss,sig_ds,sig_east,sig_north,sig_down,sig_length,sig_width,sig_strike,sig_dip,
        dist='Unif',connectivity=False,conservation=False):
        
        self.name = name
        self.ss, self.sss = ss, sig_ss
        self.ds, self.sds = ds, sig_ds  
        self.x1, self.sx1 = north, sig_north # north top-left corner
        self.x2, self.sx2 = east, sig_east # east top-left corner
        self.x3, self.sx3 = down, sig_down
        self.l, self.sl = length, sig_length
        self.w, self.sw = width, sig_width
        self.strike, self.sstrike = strike, sig_strike
        self.dip, self.sdip = dip, sig_dip
        self.dist = dist
        self.connectivity = connectivity
        self.conservation = conservation
        # initiate variable
        self.connectindex = 0
        # set uncertainties to 0 for connected patches
        if self.connectivity is not False:
            self.sstrike, self.sx3, self.sx2, self.sx1 = 0, 0, 0, 0

        # create model vector
        self.param = ['{} strike slip'.format(self.name),'{} dip slip'.format(self.name),\
        '{} north'.format(self.name),'{} east'.format(self.name),'{} down'.format(self.name),\
        '{} length'.format(self.name),'{} width'.format(self.name),'{} strike'.format(self.name),\
        '{} dip'.format(self.name)]
        
        self.m = self.tolist()
        self.sigmam = self.sigtolist()
        # self.mmin = list(map(operator.sub, self.m, self.sigmam))
        # self.mmax = list(map(operator.add, self.m, self.sigmam))

        # number of parameters per patch
        self.Mpatch = len(self.m)

        self._source = None

    def connect(self, seg):
        # set strike
        self.strike= seg.strike

        # compute vertical distance and depth
        self.x3 = seg.x3 - self.w*math.sin(np.deg2rad(self.dip))

        # compute horizontal distance
        yp = math.cos(np.deg2rad(self.dip))*self.w
        east_shift = -math.cos(np.deg2rad(seg.strike))*yp 
        north_shift = math.sin(np.deg2rad(seg.strike))*yp
        self.x2,self.x1= seg.x2+east_shift, seg.x1+north_shift

        # set uncertainties to 0
        # self.sstrike, self.sx3, self.sx2, self.sx1 = 0, 0, 0, 0

        # update m vector !!!! dangerous !!!!
        self.m = self.tolist()
        # self.sigmam = self.sigtolist()
        
    def build_prior(self):
        self.sampled = []
        self.fixed = []
        self.priors = []
        self.mmin, self.mmax =[],[]

        for name, m, sig in zip(self.param, self.m, self.sigmam):
            if sig > 0.:
                print name, m-sig, m+sig
                self.mmin.append(m-sig), self.mmax.append(m+sig)
                if self.dist == 'Normal':
                    p = pymc.Normal(name, mu=m, sd=sig)
                elif self.dist == 'Unif':
                    p = pymc.Uniform(name, lower=m-sig, upper=m+sig, value=m)
                else:
                    print('Problem with prior distribution difinition of parameter {}'.format(name))
                    sys.exit(1)
                self.sampled.append(name)
                self.priors.append(p)
            elif sig == 0:
                self.fixed.append(name)
            else:
                print('Problem with prior difinition of parameter {}'.format(name))
                sys.exit(1)
        # number of free parameters per patch
        self.Mfree = len(self.sampled)
        
    def info(self):
        print "name segment:", self.name
        src = self.get_source()
        src.regularize()
        print src
        print "#sigma_ss   sigma_ds   sigma_x1  sigma_x2  sigma_x3  sigma_length  sigma_width   sigma_strike  sigma_dip"
        print '  {:.2f}   {:.2f}   {:.1f}   {:.1f}   {:.2f}   {:.2f}   {:.2f}    {:d}     {:d}'.\
        format(*(self.sigtolist()))
        print

    def tolist(self):
        return [self.ss,self.ds,self.x1,self.x2,self.x3,self.l,
        self.w,int(self.strike),int(self.dip)]

    def sigtolist(self):
        return [self.sss,self.sds,self.sx1,self.sx2,self.sx3,self.sl,
        self.sw,int(self.sstrike),int(self.sdip)]

    def update_parameters(self, params):
        (self.ss, self.ds,
         self.x1, self.x2, self.x3,
         self.l, self.w, self.strike,
         self.dip) = map(float, params)

        src = self.source

        src.north_shift=self.x1*km
        src.east_shift=self.x2*km
        src.depth=self.x3*km
        src.width=self.w*km
        src.length=self.l*km
        src.dip=self.dip
        src.rake=np.rad2deg(math.atan2(self.ds, self.ss))
        src.strike=self.strike
        src.slip=(self.ss**2+self.ds**2)**0.5

        # self.source.regularize()
        # print self.source

        self.m = self.tolist()

    @property
    def source(self):
        if self._source is None:
            self._source = RectangularSource(
                lon=0.,  # Reference set in class inversion
                lat=0.,
                # distances in meters
                north_shift=float(self.x1*km),
                east_shift=float(self.x2*km),
                depth=float(self.x3*km),
                width=float(self.w*km),
                length=float(self.l*km),
                # angles in degree
                dip=float(self.dip),
                rake=float(np.rad2deg(math.atan2(self.ds,self.ss))), 
                strike=float(self.strike),
                slip=float((self.ss**2+self.ds**2)**0.5),
                time = self.time,
                anchor='top',
                decimation_factor=10)
        return self._source

    def get_source(self):
        return self.source

    def get_response_result(self, response, target):
        source = self.get_source()
        for src, tar, res in response.iter_results(get='results'):
            if tar is target and src is source:
                return res
        raise AttributeError('Could not find targets for %s' % manifold)

class segment:
    def __init__(self, *args, **kwargs):
        
        self.Mseg = 1
        self.prior=kwargs.pop('prior_dist', 'Unif')
        self.segments=[]

        src = patch(*args, **kwargs)

        self.segments.append(src)
        # print src.ss
        # print self.segments
        # sys.exit()

# class flower:
#     self.Mseg = 3
        





