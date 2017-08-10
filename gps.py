import numpy as np
from numpy.lib.stride_tricks import as_strided

from os import path
import sys

from pyrocko.gf import SatelliteTarget

import pymc

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
        extension='.dat',base=[0,0,0],sig_base=[0,0,0],dist='Unif'):

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
    
        # inititialisation
        self.points=[]
        self.Npoints=0
        self.N=0
        self.x,self.y=[],[]
        self.type = 'GPS' 
        self.Mbase = len(self.base)

    def load(self,inv):

        # heritated informations from flt to compute 
        # profile-parallel and profile-perp. displacements
        self.profile=inv.profile
        self.str = inv.profile.str
        self.x0 = inv.profile.x
        self.y0 = inv.profile.y

        fname=self.wdir + self.network
        if not path.isfile(fname):
            raise ValueError("invalid file name: " + fname)
        else:
            print 'Load GPS time series: ', fname
        
        f=file(fname,'r')
        # name, east(km), north(km)
        name,x,y=np.loadtxt(f,comments='#',unpack=True,dtype='S4,f,f')

        # ref to the center of the profile 
        self.xp=(x-self.x0)*self.profile.s[0]+(y-self.y0)*self.profile.s[1]
        self.yp=(x-self.x0)*self.profile.n[0]+(y-self.y0)*self.profile.n[1]

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
            station=self.wdir+self.reduction+'/'+self.name[i]+self.extension 
            print station
            if not path.isfile(station):
                raise ValueError("invalid file name: " + station)
                pass
            else:
                # print self.x[i],self.y[i],self.name[i],self.proj 
                self.points.append(gpspoint(self.x[i],self.y[i],self.name[i],self.proj))

            if 3==self.dim:
                #dated,east,north,down,esigma,nsigma,dsigma=np.loadtxt(station,comments='#',usecols=(1,2,3,4,5,6,7), unpack=True,dtype='f,f,f,f,f,f,f')
                dated,east,north,down,esigma,nsigma,dsigma=np.loadtxt(station,comments='#',usecols=(0,1,2,3,4,5,6), unpack=True,dtype='f,f,f,f,f,f,f')
                dated,east,north,down,esigma,nsigma,dsigma=np.atleast_1d(dated,east,north,down,esigma,nsigma,dsigma)
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
                self.ref = [0,0,0]
            
            if 2==self.dim:
                #date,dated,north,east,nsigma,esigma=np.loadtxt(station,comments='#',usecols=(0,1,2,3,4,5), unpack=True,dtype='f,f,f,f,f,f')
                dated,east,north,esigma,nsigma=np.loadtxt(station,comments='#',usecols=(0,1,2,3,4), unpack=True,dtype='f,f,f,f,f')
                dated,east,north,esigma,nsigma=np.atleast_1d(dated,east,north,esigma,nsigma)
                #self.points[i].d=[east,north]
                self.points[i].d=[east*self.scale,north*self.scale]
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

        self.d = np.array(self.d).flatten()
        self.sigmad = self.sigmad*np.ones(self.N)

    def info(self):
        print 'GPS time series from network:',self.network
        print 'Number of stations:', self.Npoints
        print 'Number of data:', self.N


    def g(self,inv,m):

        m = np.asarray(m)

        # forward vector
        self.gm=np.zeros((self.N))

        # have to do point by point because temporal sampling might  
        # be different for each stations
        for i in xrange(self.Npoints): 
            point = self.points[i]
            # print point.info()
            gt = as_strided(self.gm[i*self.dim*point.Nt:(i+1)*self.dim*point.Nt])
            # print 'north, east', np.ones(1)*point.y*1000., np.ones(1)*point.x*1000.

            satellite_targets = SatelliteTarget(
                north_shifts = np.ones(1)*point.y*1000.,
                east_shifts =  np.ones(1)*point.x*1000.,
                interpolation = 'nearest_neighbor',
                phi=np.ones(1),
                theta=np.ones(1))

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
                # print mpp
                # print
                mppp = np.repeat(mpp,point.Nt) 
                # print mppp
                # print np.shape(mppp), np.shape((np.ones((self.dim,point.Nt))*inv.basis[k].g(point.t)).flatten())
                
                gt += mppp*(np.ones((self.dim,point.Nt))*inv.basis[k].g(point.t)).flatten()
                # print gt
                # sys.exit()

            for k in xrange(len(inv.kernels)):
                kernel = inv.kernels[k]
                mp = as_strided(m[self.Mbase+inv.Mbasis*self.Npoints*self.dim:])
                for j in xrange(kernel.Mseg):
                    seg =  kernel.segments[j]
                    mpp = as_strided(mp[j*seg.Mpatch:(j+1)*seg.Mpatch])

                    # update patch parameter
                    seg.ss,seg.ds,seg.x1,seg.x2,seg.x3,seg.l,seg.w,seg.strike,seg.dip = mpp

                    # print seg.info()
                    # call pyrocko engine
                    disp = seg.engine(satellite_targets, inv.store, inv.store_path)
                    # print disp
                    # extract desired components
                    
                    components = ['displacement.e', 'displacement.n', 'displacement.d']

                    for ii in xrange(len(components)):
                        result = disp[components[ii]]
                        gt[point.Nt*ii:point.Nt*(ii+1)] += inv.kernels[k].g(point.t)*result

        return self.gm

    def residual(self,inv,m):
        g=np.asarray(self.g(inv,m))
        self.res = np.abs((self.d-g))/self.sigmad
        return self.res

    def jacobian(self,inv,m,epsi):
        jac=np.zeros((self.N,inv.M))
        for j in xrange(inv.M):
          # print inv.name[j]
          mp = np.copy(m)   
          mp[j] += epsi 
          jac[:,j]=(self.g(inv,mp)-self.g(inv,m))/epsi
        return jac









