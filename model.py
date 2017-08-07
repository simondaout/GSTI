import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
import matplotlib.pyplot as plt
import copy

import pymc 
# import theano
# import theano.tensor as t

# dependancies
from model import *
from flatten import *
from readgmt import *

class profile:
    def __init__(self,name='all',x=0,y=0,l=1000,w=1000,strike=0):
        # profile parameters
        self.name = name
        self.x = x
        self.y = y
        self.l = l
        self.w = w
        self.strike=strike

        # define new base 
        d2r =  math.pi/180.
        self.str = strike*d2r
        self.s = [math.sin(self.str),math.cos(self.str),0]
        self.n = [math.cos(self.str),-math.sin(self.str),0]

        # define boundaries profile
        self.ypmax,self.ypmin = self.l/2,self.l/2
        self.xpmax,self.xpmin = self.w/2,self.w/2

class inversion:
    def __init__(self,kernels,basis,manifolds,profile,gmtfiles,store_path='./',store=None,bounds=None):
        
        self.kernels=flatten(kernels)
        self.basis=flatten(basis)
        self.manifolds=manifolds
        self.profile = profile
        self.store_path=store_path
        self.store=store
        self.gmtfiles=gmtfiles
        self.bnds=bounds

        self.Mker = len(self.kernels)
        self.Mbasis = len(self.basis)
        self.Mmanif = len(self.manifolds)

        # load data and build model vector for each manifolds
        for i in xrange(self.Mmanif):
            self.manifolds[i].load(self)
            self.manifolds[i].info()

        # Careful: after loading data
        self.N = sum(map((lambda x: getattr(x,'N')),self.manifolds))
        self.Npoints = sum(map((lambda x: getattr(x,'Npoints')),self.manifolds))

        # get all segments
        segments = []
        for k in xrange(self.Mker):
            segments.append(map((lambda x: getattr(x,'segments')),self.kernels[k].structures))
        self.segments = flatten(segments)
        self.Mseg = len(self.segments)        
        # print self.segments
        # sys.exit()

        # # free parameters
        # self.Mpatch = sum(map((lambda x: getattr(x,'Mpatch')),self.segments))

    def info(self):
        print
        print 'Spacial functions: '
        print '#index #inversion_type #name #date #nb_structures #nb_segments'
        for i in xrange(len(self.kernels)):
            kernel=self.kernels[i]
            print '{0:n} {1:s} {2:s} {3:4.1f} {4:n} {5:n}'.format(i, kernel.inversion_type,\
            kernel.name, kernel.date, kernel.Mstr, kernel.Mseg)
        print

        print 'Time functions: '
        print '#index #inversion #name #date'
        for i in xrange(len(self.basis)):
            basis=self.basis[i]
            print '{0:n} {1:s} {2:s} {3:4.1f}'.format(i, basis.inversion_type,\
            basis.name, basis.date)
        print

    def build_data(self):
        self.d = np.zeros((self.N))
        # print self.N
        start = 0
        for i in xrange(self.Mmanif):
            # print self.manifolds[i].N, len(self.manifolds[i].d)
            self.d[start:start+self.manifolds[i].N] = self.manifolds[i].d
            start+=self.manifolds[i].N
        return self.d

    def build_prior(self):

        # vector m = [ [m_ref, [m_basis for each point and each dim] for each manifolds],
        # m_faults for each segments ]

        self.priors = []
        self.sampled = [] 
        self.fixed = []
        self.minit = []
        self.name = []
        self.mmin, self.mmax = [], []

        for i in xrange(self.Mmanif):
            manifold = self.manifolds[i]
            # Baseline 
            for ii in xrange(self.manifolds[i].Mbase):
                name = '{} Baseline {}'.format(manifold.reduction,ii)
                self.minit.append(manifold.base[ii])
                self.mmin.append(manifold.base[ii]-manifold.sig_base[ii])
                self.mmax.append(manifold.base[ii]+manifold.sig_base[ii])
                self.name.append(name)
                
                if manifold.sig_base[ii] > 0.:
                    m, sig = manifold.base[ii], manifold.sig_base[ii]
                    if self.dist == 'Normal':
                        p = pymc.Normal(name, mu=m, sd=sig)
                    elif self.dist == 'Unif':
                        p = pymc.Uniform(name, lower=m-sig, upper=m+sig)
                    else:
                        print('Problem with prior distribution difinition of parameter {}'.format(name))
                        sys.exit(1)
                    self.sampled.append(name)
                    self.priors.append(p)

                elif manifold.sig_base[ii] == 0:
                    self.fixed.append(name)

                else:
                    print('Problem with prior difinition of parameter {}'.format(name))
                    sys.exit(1)

            # Basis functions parameter for each dim of each manifolds
            for k in xrange(self.Mbasis):

                # name = 'basis:{}'.format(self.basis[k].name)
                # if self.basis[k].sigmam > 0.:
                #     p = pymc.Uniform(name, lower=m-sig, upper=m+sig, shape=manifold.Npoints*manifold.dim)
                #     self.sampled.append(name)
                #     self.priors.append(p)
                # elif self.basis[k].sigmam == 0:
                #     self.fixed.append(name)

                for j in xrange(manifold.Npoints):
                    point = manifold.points[j]

                    for ii in xrange(manifold.dim):

                        name = 'Point:{}{}, dim:{}, basis:{}'.format(point.name,j,ii,self.basis[k].name)
                        self.minit.append(self.basis[k].m)
                        self.mmin.append(self.basis[k].m-self.basis[k].sigmam)
                        self.mmax.append(self.basis[k].m+self.basis[k].sigmam)
                        self.name.append(name)
                       
                        # print self.basis[k].sigmam
                        if self.basis[k].sigmam > 0.:
                            
                            m, sig = self.basis[k].m, self.basis[k].sigmam
                            # print name, m, sig
                            if self.basis[k].dist == 'Normal':
                                p = pymc.Normal(name, mu=m, sd=sig)
                            elif self.basis[k].dist == 'Unif':
                                p = pymc.Uniform(name, lower=m-sig, upper=m+sig)
                            else:
                                print('Problem with prior distribution difinition of parameter {}'.format(name))
                                sys.exit(1)
                            self.sampled.append(name)
                            self.priors.append(p)

                        elif self.basis[k].sigmam == 0:
                            self.fixed.append(name)

                        else:
                            print('Problem with prior difinition of parameter {}'.format(name))
                            sys.exit(1)

        # number of basis parameters
        self.Msurface = len(np.array(flatten(self.minit)))

        # Faults parameters
        self.fixed.append(map((lambda x: getattr(x,'fixed')),self.segments))
    
        list_of_lists=map((lambda x: getattr(x,'priors')),self.segments)
        flattened_list = [y for x in list_of_lists for y in x]
        for item in flattened_list:
            self.priors.append(item)

        list_of_lists=map((lambda x: getattr(x,'mmin')),self.segments)
        flattened_list = [y for x in list_of_lists for y in x]
        for item in flattened_list:
            self.mmin.append(item)

        list_of_lists=map((lambda x: getattr(x,'mmax')),self.segments)
        flattened_list = [y for x in list_of_lists for y in x]
        for item in flattened_list:
            self.mmax.append(item)
        
        # self.mmin.append(map((lambda x: getattr(x,'mmin')),self.segments))
        # self.mmax.append(map((lambda x: getattr(x,'mmax')),self.segments))

        # self.priors.append(map((lambda x: getattr(x,'priors')),self.segments))
        self.sampled.append(map((lambda x: getattr(x,'sampled')),self.segments))
        self.minit.append(map((lambda x: getattr(x,'m')),self.segments))
        self.name.append(map((lambda x: getattr(x,'param')),self.segments))

        # print
        # convert to array
        self.fixed = np.array(flatten(self.fixed))
        # print 'fixed:', self.fixed 
        self.priors = np.array(self.priors).flatten()
        # print 'priors:', self.priors
        # sys.exit()
        self.sampled =  np.array(flatten(self.sampled))
        # print 'sampled:', self.sampled
        self.minit =  np.array(flatten(self.minit))
        # print 'minit:', self.minit
        self.name =  np.array(flatten(self.name))
        # print 'name:', self.name
        # print

        # self.minit, self.mmin, self.mmax = np.array(self.minit),np.array(self.mmin),np.array(self.mmax)

        # initialize m
        self.m = np.copy(self.minit)
        # sys.exit()

        return self.minit

    def build_gm(self,theta):
        g = np.zeros((self.N))
        start=0

        M = 0
        for i in xrange(self.Mmanif):
            manifold = self.manifolds[i]
            index = manifold.Mbase+self.Mbasis*manifold.Npoints*manifold.dim

            # print theta[:self.Msurface]
            # sys.exit()
            mp = as_strided(theta[M:M+index])
            mpp = as_strided(theta[self.Msurface:])

            m = np.concatenate([mp,mpp])

            g[start:start+manifold.N]=manifold.g(self,m)
            
            start+=manifold.N
            M += index

        return g

    # @theano.compile.ops.as_op(itypes=[t.lscalar],otypes=[t.dvector])
    def foward(self, theta):
        msampled = theta
        g = np.zeros((self.N))

        # Rebuild the full m vector
        self.m = []
        uu = 0
        # print len(self.minit)
        for name, initial in zip(self.name, self.minit):
            # print self.sampled
            # sys.exit()
            
            if name in self.sampled:
                # print uu, name
                # print msampled[uu]
                self.m.append(msampled[uu])
                uu +=  1
            elif name in self.fixed:
                # print initial
                # print initial.dtype
                self.m.append(initial)

        # print np.array(self.m[:self.Msurface])
        # print self.m[self.Msurface:] 
        # sys.exit()

        self.m = np.array(self.m)
        start=0
        M=0
        for i in xrange(self.Mmanif):
            manifold = self.manifolds[i]
            index = manifold.Mbase+self.Mbasis*manifold.Npoints*manifold.dim

            mp = as_strided(self.m[M:M+index])
            mpp = as_strided(self.m[self.Msurface:])
            m = np.concatenate([mp,mpp])

            g[start:start+manifold.N]=manifold.g(self,m)
            
            start+=manifold.N
            M += index
        return g

    def residual(self,theta):
        r=np.zeros((self.N))
        start=0
        M=0
        for i in xrange(self.Mmanif):
            manifold = self.manifolds[i]
            index = manifold.Mbase+self.Mbasis*manifold.Npoints*manifold.dim
            
            # rebuild m for each manifold
            index = manifold.Mbase+self.Mbasis*manifold.Npoints*manifold.dim
            mp = as_strided(theta[M:M+index])
            mpp = as_strided(theta[self.Msurface:])
            m = np.concatenate([mp,mpp])

            r[start:start+manifold.N]=manifold.residual(self,m)
            start+=manifold.N
            M += index
        return r

    def residualscalar(self,theta):
        res = np.sum(pow(self.residual(theta),2))
        # print res
        return res

    def Cov(self):
        Cov = np.zeros((self.N))
        start = 0
        for i in xrange(len(self.manifolds)):
            # # print self.manifolds[i].sigmad
            # Cd = np.diag(self.manifolds[i].sigmad**2,k = 0)
            # # And the diagonal of its inverse
            # Cov[start:start+self.manifolds[i].N] = np.diag(np.linalg.inv(Cd))
            # # Save Covariance matrix for each data set
            # manifolds[i].Cd = np.diag(Cd)
            # manifolds[i].invCd = np.diag(np.linalg.inv(Cd))

            Cov[start:start+self.manifolds[i].N] = 1/self.manifolds[i].sigmad**2

            start+= self.manifolds[i].N
        return Cov

    def plot_ts_GPS(self):
        for n in xrange(self.Mmanif):
            manifold = self.manifolds[n]
            if manifold.type=='GPS':

                for i in xrange(manifold.Npoints):
                    point=manifold.points[i]
                    gt = as_strided(manifold.gm[i*manifold.dim*point.Nt:(i+1)*manifold.dim*point.Nt])
                    rest = as_strided(manifold.res[i*manifold.dim*point.Nt:(i+1)*manifold.dim*point.Nt])

                    fig=plt.figure(figsize=(18,7))
                    fig.subplots_adjust(hspace=0.7)

                    for j in xrange((manifold.dim)):
                        ax1 = fig.add_subplot(3,manifold.dim,j+1)

                        lig = ax1.plot(point.t,point.d[j],'.',label='{} {}'.format(point.name, manifold.plot[j]))
                        # ax1.errorbar(point.t,point.d[j],yerr=point.sigmad[j],label=manifold.plot[j],color=lig[0].get_color())
                        ax1.plot(point.t,gt[point.Nt*j:point.Nt*(j+1)],color='black',linewidth=4.0)
                        ax1.legend(bbox_to_anchor=(0.,1.02,1,0.102),loc=3,ncol=2,mode='expand',borderaxespad=0.)
                        if j==0:
                            ax1.set_ylabel('Displacements (m)')
                        # ax1.set_xlabel('Time')
                        ax1.grid(True)
                        ax1.set_xlim([point.tmin,point.tmax])
                        locs,labels = plt.xticks()
                        ax1.set_xticks(locs, map(lambda x: "%g" %x, locs))
                        ax1.set_autoscalex_on(False)

                        ax2 = fig.add_subplot(3,manifold.dim,manifold.dim+j+1)
                        # ax2.plot(point.t,manifold.residual(self)[point.Nt*j:point.Nt*(j+1)])
                        ax2.plot(point.t,rest[point.Nt*j:point.Nt*(j+1)])
                        if j==0:
                            ax2.set_ylabel('Residuals (m)')
                        # ax2.set_xlabel('Time')
                        ax2.grid(True)
                        locs,labels = plt.xticks()
                        plt.xticks(locs, map(lambda x: "%g" %x, locs))
                        ax2.set_xlim([point.tmin,point.tmax])
                        ax2.set_autoscalex_on(False)

                        ax3 = fig.add_subplot(3,manifold.dim,2*manifold.dim+j+1)
                        
                        for k in xrange(0,self.Mbasis):
                            gm = np.ones((point.Nt))*self.basis[k].g(point.t)
                            ax3.plot(point.t,gm,label=self.basis[k].name)
                        if j==0:
                            ax3.set_ylabel('Temporal functions (m)')
                        ax3.set_xlabel('Time')
                        ax3.grid(True)
                        ax3.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3,ncol=2,mode='expand',borderaxespad=0.)
                        locs,labels = plt.xticks()
                        plt.xticks(locs, map(lambda x: "%g" %x, locs))
                        ax3.set_xlim([point.tmin,point.tmax])
                        ax3.set_autoscalex_on(False)
                            

                # fig.autofmt_xdate()
                plt.title('Station {} time series'.format(point.name))
                # fig.tight_layout()

    def plot_InSAR_maps(self):
        for n in xrange(self.Mmanif):
            manifold = self.manifolds[n]

            if manifold.type=='InSAR':
                fig, _ = plt.subplots(1,3,figsize=(12,4))

                vranges = [(manifold.gm[1::2].max(),
                    manifold.gm[1::2].min())]

                lmax = np.abs([np.min(vranges), np.max(vranges)]).max()
                levels = np.linspace(-lmax, lmax, 50)

                ax = fig.axes[0]
                cmap = ax.tricontourf(manifold.x, manifold.y, manifold.d[1::2],
                                cmap='seismic', levels=levels)

                plotgmt(self.gmtfiles, ax)

                if self.bnds is not None:
                    ax.set_xlim(self.bnds[0],self.bnds[1])
                    ax.set_ylim(self.bnds[2],self.bnds[3])

                ax.set_title('Data: {}'.format(manifold.reduction))
                ax.set_aspect('equal')
                ax.set_xlabel('[km]')
                ax.set_ylabel('[km]')

                fig.colorbar(cmap, ax=ax, aspect=5)

                # plot gps stations
                for nn in xrange(self.Mmanif):
                  if self.manifolds[nn].type=='GPS':
                    for ii in xrange(self.manifolds[nn].Npoints):
                        gps = self.manifolds[nn].points[ii]
                        ax.scatter(gps.x, gps.y, c = 'black', s = 40, marker = '^')
                        ax.text(gps.x+1,gps.y+1,gps.name,color='black',fontsize='x-small')


                ax = fig.axes[1]
                cmap = ax.tricontourf(manifold.x, manifold.y, manifold.gm[1::2],
                                cmap='seismic', levels=levels)

                ax.set_title('Model')
                ax.set_aspect('equal')
                ax.set_xlabel('[km]')
                ax.set_ylabel('[km]')

                for seg in self.segments:
                    # fe,fn = source.outline.T
                    # ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
                    fn, fe = seg.source.outline(cs='xy').T/1000
                    ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
                    ax.plot(fe[:2],fn[:2],linewidth=2.,color='black',alpha=0.5)

                plotgmt(self.gmtfiles, ax)

                if self.bnds is not None:
                    ax.set_xlim(self.bnds[0],self.bnds[1])
                    ax.set_ylim(self.bnds[2],self.bnds[3])

                fig.colorbar(cmap, ax=ax, aspect=5)

                ax = fig.axes[2]
                cmap = ax.tricontourf(manifold.x, manifold.y, manifold.res[1::2],
                                cmap='seismic', levels=levels)

                plotgmt(self.gmtfiles, ax)

                ax.set_title('Residual')
                ax.set_aspect('equal')
                ax.set_xlabel('[km]')
                ax.set_ylabel('[km]')

                if self.bnds is not None:
                    ax.set_xlim(self.bnds[0],self.bnds[1])
                    ax.set_ylim(self.bnds[2],self.bnds[3])

                fig.colorbar(cmap, ax=ax, aspect=5)

                fig.tight_layout()

def plotgmt(gmtfiles, ax):
    for ii in xrange(len(gmtfiles)):
        name = gmtfiles[ii].name
        wdir = gmtfiles[ii].wdir
        filename = gmtfiles[ii].filename
        color = gmtfiles[ii].color
        width = gmtfiles[ii].width
        fx,fy = gmtfiles[ii].load()
        for i in xrange(len(fx)):
            ax.plot(fx[i],fy[i],color = color,lw = width)
















