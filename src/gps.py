import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided

from os import path
import sys

from pyrocko.gf import StaticTarget

import pymc

km = 1000.

logger = logging.getLogger('GSTI.gps')


class point:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class gpspoint(point):
    def __init__(self,x,y,name,proj):
        point.__init__(self,x,y)
        self.name=name
        self.t=[]
        self.d=[] #East,North,Down
        self.sigmad=[] #East,North, Down
        self.proj=proj

        # lenght of the time series
        self.Nt = 0

    def info(self):
        print 'gps station: {}, east(km): {}, north(km): {}'.format(self.name,self.x,self.y)

class gpstimeseries:
    def __init__(self,network,reduction,dim,wdir,scale=1.,weight=1.,proj=[1.,1.,1.],
        extension='.dat',base=[0,0,0],sig_base=[0,0,0],dist='Unif',store_id=None):

        self.network=network
        self.reduction=reduction
        self.dim=dim
        self.wdir=wdir
        self.scale=scale
        self.sigmad=1./weight
        self.proj=proj
        self.extension=extension
        self.base=base
        self.sig_base=sig_base
        self.dist=dist
        self.store_id=store_id
    
        # inititialisation
        self.points=[]
        self.Npoints=0
        self.N=0
        self.x,self.y=[],[]
        self.type = 'GPS' 
        self.Mbase = len(self.base)

        self._targets = None

    def load(self,inv):

        # heritated informations from flt to compute 
        # profile-parallel and profile-perp. displacements
        # self.profile=inv.profile
        # self.str = inv.profile.str
        # self.x0 = inv.profile.x
        # self.y0 = inv.profile.y

        fname=self.wdir + self.network
        if not path.isfile(fname):
            raise ValueError("invalid file name: " + fname)
        else:
            print 'Load GPS time series: ', fname
            print 
        
        f = file(fname, 'r')
        # name, east(km), north(km)
        name,x,y=np.loadtxt(f,comments='#',unpack=True,dtype='S4,f,f')

        # ref to the center of the profile 
        # self.xp=(x-self.x0)*self.profile.s[0]+(y-self.y0)*self.profile.s[1]
        # self.yp=(x-self.x0)*self.profile.n[0]+(y-self.y0)*self.profile.n[1]

        # # select points within profile
        # index=np.nonzero((xp>self.profile.xpmax)|(xp<self.profile.xpmin)|(yp>self.profile.ypmax)|(yp<self.profile.ypmin))
        # self.name,self.x,self.y,self.xp,self.yp=np.delete(name,index),np.delete(x,index),\
        # np.delete(y,index),np.delete(xp,index),np.delete(yp,index)

        self.name,self.x,self.y=np.atleast_1d(name,x,y)
        # print self.name
        self.Npoints=len(self.name)
        self.N=0
        self.d = []
        
        print 'Load time series... '
        for i in xrange(self.Npoints):
            fn_station = path.join(self.wdir, self.reduction, self.name[i] + self.extension)
            fn_station = path.abspath(fn_station)
            if not path.isfile(fn_station):
                raise ValueError("invalid file name: " + fn_station)
            else:
                # print self.x[i],self.y[i],self.name[i],self.proj 
                self.points.append(gpspoint(self.x[i],self.y[i],self.name[i],self.proj))
                print self.x[i],self.y[i],self.name[i],self.proj
            if 3==self.dim:
                print(fn_station)
                #dated,east,north,down,esigma,nsigma,dsigma=np.loadtxt(fn_station,comments='#',usecols=(1,2,3,4,5,6,7), unpack=True,dtype='f,f,f,f,f,f,f')
                data = np.loadtxt(fn_station, unpack=True)
                dated,east,north,down,esigma,nsigma,dsigma = np.atleast_1d(data)

                # print  dated,east,north,down,esigma,nsigma,dsigma

                self.points[i].d=[east*self.scale,north*self.scale,down*self.scale]
                self.points[i].sigmad=[esigma*self.sigmad*self.scale,nsigma*self.sigmad*self.scale,dsigma*self.sigmad*self.scale]
                #self.points[i].sigmad=[self.sigmad*self.scale,self.sigmad*self.scale,self.sigmad*self.scale]
                self.plot=['east','north','down']
                self.points[i].t=np.atleast_1d(dated)
                self.points[i].veast=east[-1]-east[0]/(dated[-1]-dated[0])
                self.points[i].vnorth=north[-1]-north[0]/(dated[-1]-dated[0])
                self.points[i].vdown=down[-1]-down[0]/(dated[-1]-dated[0])
                # print north[-1], north[0], dated[-1], dated[0]
                # print self.points[i].vnorth, self.points[i].veast

                # reference frame
                self.ref = [0, 0, 0]
            
            if 2==self.dim:
                #date,dated,north,east,nsigma,esigma=np.loadtxt(fn_station,comments='#',usecols=(0,1,2,3,4,5), unpack=True,dtype='f,f,f,f,f,f')
                dated,east,north,esigma,nsigma=np.loadtxt(fn_station,comments='#',usecols=(0,1,2,3,4), unpack=True,dtype='f,f,f,f,f')
                dated,east,north,esigma,nsigma=np.atleast_1d(dated,east,north,esigma,nsigma)
                #self.points[i].d=[east,north]
                self.points[i].d=[east*self.scale, north*self.scale]
                self.points[i].sigmad=[esigma*self.sigmad*self.scale,nsigma*self.sigmad*self.scale]
                #self.points[i].sigmad=[self.sigmad*self.scale,self.sigmad*self.scale]
                self.plot=['east','north']
                self.points[i].t=np.atleast_1d(dated)
                self.points[i].veast=east[-1]-east[0]/(dated[-1]-dated[0])
                self.points[i].vnorth=north[-1]-north[0]/(dated[-1]-dated[0])

                # reference frame
                self.ref = [0,0]

            self.points[i].tmin,self.points[i].tmax=min(dated),max(dated)
            if len(dated)==1:
                self.points[i].tmin=self.points[i].tmax-1.
                #self.points[i].tmin,self.points[i].tmax=1992.,2004.
            
            self.points[i].Nt=len(self.points[i].t)
            self.N += len(self.points[i].t)*self.dim
            self.d.append(self.points[i].d)
            self.points[i].lon = inv.ref[0]
            self.points[i].lat = inv.ref[1]

        self.d = np.array(self.d).flatten()
        self.sigmad = self.sigmad*np.ones(self.N)

    def info(self):
        print
        print 'GPS time series from network:',self.network
        print 'Number of stations:', self.Npoints
        print 'Lenght data vector:', self.N

    def printbase(self):
        print 'Reference frame:', self.base
        print 

    def get_targets(self):
        if self._targets is None:
            self._targets = StaticTarget(
                lats=np.array([p.lat for p in self.points]),
                lons=np.array([p.lon for p in self.points]),
                north_shifts=np.array([p.y for p in self.points]),
                east_shifts=np.array([p.x for p in self.points]),
                interpolation='nearest_neighbor',
                store_id=self.store_id)
        return self._targets

    def g(self, inv, m, response):
        logger.debug('Calculating G Matrix...')
        m = np.asarray(m)

        # forward vector
        self.gm=np.zeros((self.N))

        # have to do point by point because temporal sampling might  
        # be different for each stations

        temp = 0
        for i in xrange(self.Npoints): 
            point = self.points[i]
            # print point.info()
            gt = as_strided(self.gm[temp:temp+self.dim*point.Nt])
            # print 'north, east', np.ones(1)*point.y*km, np.ones(1)*point.x*km


            # update reference frame
            self.base = m[:self.Mbase]
            ref = np.ones((self.dim,point.Nt))
            # translation vector
            for ii in xrange(self.dim):
                ref[ii,:] = self.base[ii]

            # print ref.flatten()
            # sys.exit()
            # print np.shape(gt), np.shape(ref.flatten())
            gt += ref.flatten()

            for k in xrange(len(inv.basis)): 
                mp = as_strided(m[self.Mbase+k*self.Npoints*self.dim:self.Mbase+(k+1)*self.Npoints*self.dim])
                mpp = as_strided(mp[i*self.dim:(i+1)*self.dim])
                mppp = np.repeat(mpp,point.Nt) 

                gt += mppp*(np.ones((self.dim,point.Nt))*inv.basis[k].g(point.t)).flatten()

            for kernel in inv.kernels:
                for seg in kernel.segments:
                    resp = seg.get_response_result(response, self.get_targets())
                    disp = resp.result
                    # print disp
                    # extract desired components
                    
                    components = ['displacement.e', 'displacement.n', 'displacement.d']

                    for ii in xrange(len(components)):
                        # print components[ii]
                        result = disp[components[ii]][i]
                        gt[point.Nt*ii:point.Nt*(ii+1)] += kernel.g(point.t)*result
                        # print inv.kernels[k].g(point.t)
                        # print result
                        # print

                    
            temp += self.dim*point.Nt

        # sys.exit()
        return self.gm

    def residual(self, inv, m, response):
        g=np.asarray(self.g(inv, m, response))
        self.res = (self.d-g)/self.sigmad
        return self.res

    def jacobian(self,inv,m,epsi):
        jac=np.zeros((self.N,inv.M))
        for j in xrange(inv.M):
          # print inv.name[j]
          mp = np.copy(m)   
          mp[j] += epsi 
          jac[:,j]=(self.g(inv,mp)-self.g(inv,m))/epsi
        return jac

