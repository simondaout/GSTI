import numpy as np
from numpy.lib.stride_tricks import as_strided
import pymc

from os import path
import math, sys

from pyrocko.gf import SatelliteTarget

class point:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class insarpoint(point):
    def __init__(self,x,y,name,los,proj,tmin,tmax,sigmad):
        point.__init__(self,x,y)
        self.name=name
        self.proj=proj #East,North,Down
        self.d=[[los]]
        self.sigmad=[[sigmad]]
        self.tmin=tmin
        self.tmax=tmax

        self.t=np.atleast_1d([tmax-tmin])

        # length of the time series
        self.Nt=2

class insarstack:
    def __init__(self,network,reduction,wdir,proj=None,los=None,heading=None,
        tmin=0.,tmax=1.,weight=1.,scale=1.,base=[0,0,0],sig_base=[0,0,0],dist='Unif'):

        self.network=network
        self.reduction=reduction
        self.wdir=wdir
        self.los=los
        self.heading=heading
        self.tmin=tmin
        self.tmax=tmax
        self.sigmad=1./weight
        self.scale=scale
        self.projm=proj
        self.base=base
        self.sig_base=sig_base
        self.dist=dist

        self.points=[]
        self.plot='LOS'
        self.type = 'InSAR'
        self.N=0
        self.Npoints=0
        self.dim=1
        self.Mbase = len(self.base)

    def load(self,inv):
        fname=self.wdir + self.network
        if not path.isfile(fname):
            raise ValueError("invalid file name: " + fname)
        else:
            print 'Load InSAR stack: ', fname
            
        # heritated informations from flt to compute 
        # profile-parallel and profile-perp. displacements
        self.profile=inv.profile
        self.str = inv.profile.str
        self.x0 = inv.profile.x
        self.y0 = inv.profile.y

        f=file(fname,'r')
        if self.los is not None:
            # load x, y, los, incidence angle 
            x,y,los,theta=np.loadtxt(f,comments='#',unpack=True,dtype='f,f,f,f')

            # ref to the center of the profile 
            xp=(x-self.x0)*self.profile.s[0]+(y-self.y0)*self.profile.s[1]
            yp=(x-self.x0)*self.profile.n[0]+(y-self.y0)*self.profile.n[1]
            
            # select point within profile
            index=np.nonzero((xp>self.profile.xpmax)|(xp<self.profile.xpmin)|(yp>self.profile.ypmax)|(yp<self.profile.ypmin))
            self.ulos,self.x,self.y,self.xp,self.yp,self.theta=np.delete(los,index),np.delete(x,index),np.delete(y,index),np.delete(xp,index),np.delete(yp,index),np.delete(theta,index)
            
            # compute phi, theta, pyrocko convention
            self.phi = np.deg2rad(-90-self.heading)
            self.theta = np.deg2rad(90.-self.theta)
            
            # check phim, thetam
            self.phim,self.thetam=np.mean(self.phi),np.mean(self.theta)
            
            # compute proj vector
            np.array([
            np.cos(self.theta)*np.cos(self.phi),
            np.cos(self.theta)*np.sin(self.phi),
            np.sin(self.theta)
            ]).T

            # compute average proj vector for GPS data
            self.projm=[
            np.cos(thetam)*np.cos(phim),
            np.cos(thetam)*np.sin(phim),
            np.sin(thetam)]

        else:
            # load x, y, los
            self.x,self.y,self.ulos=np.loadtxt(f,comments='#',unpack=True,dtype='f,f,f')
            
            # ref to the center of the profile 
            self.xp=(self.x-self.x0)*self.profile.s[0]+(self.y-self.y0)*self.profile.s[1]
            self.yp=(self.x-self.x0)*self.profile.n[0]+(self.y-self.y0)*self.profile.n[1]

            # select point within profile
            # index=np.nonzero((xp>self.profile.xpmax)|(xp<self.profile.xpmin)|(yp>self.profile.ypmax)|(yp<self.profile.ypmin))
            # self.ulos,self.x,self.y,self.xp,self.yp=np.delete(los,index),np.delete(x,index),np.delete(y,index),np.delete(xp,index),np.delete(yp,index)

            if self.projm is not None:
                self.proj = np.array([self.projm]*len(self.ulos))
            else:
                print 'No average projection look angle set in the profile class'
                sys.exit(2)

            self.phim = math.atan2(self.projm[1],self.projm[0])
            self.thetam = math.atan2(self.projm[2],(self.projm[1]**2+self.projm[0]**2)**.5)
            # sys.exit()

            self.phi,self.theta = np.repeat(self.phim,len(self.ulos)), np.repeat(self.thetam,len(self.ulos))
            
        
        # vector data: [los1_date1, los1_date2, los2_date1, los2_date2 ,..., losNpoints_date1, losNpoints_date2]
        self.N, self.Npoints = len(self.ulos)*2, len(self.ulos)
        self.d = np.zeros((self.N))
        self.d[1::2] = self.ulos
        self.sigmad = np.ones((self.N))*self.sigmad
        self.t = np.tile([self.tmin, self.tmax],self.Npoints)
        self.Nt = 2

        for i in xrange(self.Npoints):
            self.points.append(insarpoint(self.x[i],self.y[i],self.reduction,self.scale*self.ulos[i]*(self.tmax-self.tmin),\
                self.proj[i],self.tmin,self.tmax,self.sigmad))

    # create orbital function
    def reference(self):
          func = self.base[0]*self.x + self.base[1]*self.y + self.base[2]
          return  func


    def info(self):
        print 'InSAR map acquiered between tmin:{} and tmax:{}'.format(self.tmin,self.tmax)
        print 'Number of points:', self.Npoints
        print 'phi: {}, theta: {}'.format(np.rad2deg(self.phim),np.rad2deg(self.thetam))
        print '[East, North, Down] vector :', self.projm
        # print self.d[:20]
        # sys.exit()
        print


    def g(self,inv,m):

        m = np.asarray(m)
        # print m
        # m = m.astype(float)

        # forward vector
        self.gm=np.zeros((self.N))

        # print self.x,self.y

        satellite_targets = SatelliteTarget(
            # distance in meters
            # These changes of units are shit !
            north_shifts = self.y*1000.,
            east_shifts =  self.x*1000.,
            interpolation='nearest_neighbor',
            # phi, theta in rad
            phi=self.phi,
            theta=self.theta)

        # update reference brame
        self.base = m[:self.Mbase]
        # print self.base
        self.gm[1::2] += self.reference()  
        # print self.gm      

        # for k in xrange(inv.Mbasis):
        #     # one m value for each basis and each point
        #     # duplicate m for all times
        #     mp = as_strided(m[self.Mbase+k*self.Npoints:self.Mbase+(k+1)*self.Npoints])
        #     # print mp
        #     mpp = np.repeat(mp,self.Nt)
        #     # print mpp.dtype
        #     # print inv.basis[k].g(self.t).dtype
        #     # print

        #     self.gm += mpp*inv.basis[k].g(self.t)

        start = 0
        for k in xrange(len(inv.kernels)):
            kernel = inv.kernels[k]
            # print kernel.name
            # mp = as_strided(m[self.Mbase+inv.Mbasis*self.Npoints:])
            mp = as_strided(m[self.Mbase:])
            for j in xrange(kernel.Mseg):
                seg =  kernel.segments[j]
                mpp = as_strided(mp[start:start+seg.Mpatch])

                # update patch parameter
                seg.ss,seg.ds,seg.x1,seg.x2,seg.x3,seg.l,seg.w,seg.strike,seg.dip = mpp

                # call pyrocko engine and extract los component
                disp = seg.engine(satellite_targets, inv.store, inv.store_path)['displacement.los']
                # print disp

                # gt = np.zeros((self.N))
                gt = np.repeat(disp,2)
                # print gt

                self.gm += inv.kernels[k].g(self.t)*gt
                # print self.t
                # print inv.kernels[k].g(self.t)
                # print self.gm
                # print
                start += seg.Mpatch 

                # print self.t
                # print inv.kernels[k].g(self.t)
        # sys.exit()

        return self.gm

    def residual(self,inv,m):
        g=np.asarray(self.g(inv,m))
        # print
        # print g
        # print self.d
        # # print self.d-g
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










