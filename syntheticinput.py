#!/usr/bin/env python2.7
import numpy as num
import os
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc

# GSti dependencies
from kernel import *
from model import *
from gps import *
from insar import *
from structures import *
from readgmt import *
from combisource import *
from date2dec import *

# pyrcoko dependencies
from pyrocko.gf import LocalEngine, StaticTarget, SatelliteTarget,\
    RectangularSource

# define the Green Function store for the synthetic example
store='halfspace'
store_path=['./']

#####################################################
############ CREATE SYNTHETIC EXAMPLE ###############
#####################################################

engine = LocalEngine(store_superdirs=store_path,default_store_id=store)

##############################################
#            Targets                         #
##############################################

# distance in kilometer
km = 1e3

# We define a grid for the targets.
left,right,bottom,top=-35,15,-25,25,

# Synthetic GPS points 
# stations_name = [   'XIAO']
# stations_east    = [-15104.34]
# stations_north    = [-22552.38]
stations_name = [   'XIAO',    'A01',  'A02',   'A03',   'A04',   'A05' ]
stations_east    = [-15104.34, -100000.,     0.,      0.,  10000.,  10000. ]
stations_north    = [-22552.38,  100000., 10000., -10000., -10000.,  10000. ]
Ngps=len(stations_name)
# print stations_north[0], stations_east[0]

# Synthetic InSAR points
Ninsar = 500
# Ninsar = 10000
# caracteristic of the Envisat satellite
heading=-76.
look=20.5
phi = num.empty(Ninsar+Ngps) # Horizontal LOS from E in anti-clokwise rotation
theta = num.empty(Ninsar+Ngps)  # Vertical LOS from horizontal
phi.fill(num.deg2rad(-90-heading)) # Carefull in rad.
theta.fill(num.deg2rad(90.-look))

satellite_target = SatelliteTarget(
    north_shifts = np.concatenate(np.array([rnd.uniform(bottom*km, top*km, Ninsar),stations_north])),
    east_shifts= np.concatenate(np.array([rnd.uniform(left*km, right*km, Ninsar),stations_east])),
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta)

# Define the [east, north, down] average projection vector
projm=[num.cos(theta[0])*num.cos(phi[0]),
      num.cos(theta[0])*num.sin(phi[0]),
      num.sin(theta[0])
     ]

##############################################
#            Create coseismic model          #
##############################################

def mw2slip(mw,l,W):
    m0 = 10**((mw+6.07)*3./2)
    potency=m0/31.5e9
    slip=potency/(l*km*W*km)
    # print 'slip:{}'.format(slip)
    return slip

# 2009 event
# GCMT: lon=95.76, lat=37.64, Depth:12km, strike=101, dip:60, rake:83, mw=6.3
# [Elliott 2011]: Depth:4.7km, strike=100, dip:53, rake:106, length=12.2km, width=5.4km, mw=6.3
# [Feng 2015]:  Depth:5km, strike=108, dip:53, rake:90, length=xkm, width=xkm, mw=6.3

y,x=-12.5,4
d = 5.
strike=108
dip=53.
rake = 90 
l = 12.
W=5.5
mw=6.3
slip = mw2slip(mw,l,W)

coseis = RectangularSource(
    north_shift=x*km, east_shift=y*km,
    depth=d*km, width=W*km, length=l*km,
    dip=dip, rake=rake, strike=strike,
    slip=slip)

print
print 'Synthetic model:'
print coseis

patches = [coseis];
sources = CombiSource(subsources=patches)

# The computation is performed by calling process on the engine
result = engine.process(sources, [satellite_target])

##############################################
#       Create synthetic time series         #
##############################################

# convert some dates to decimal time
dates = [20090828, 20090708, 20091021]
# dates = [20090828, 20080110, 20100421]
times = date2dec(dates)
# print times
# sys.exit()

# define time for synthetic time series 
t0 = 2005.
t = t0+num.arange(0,8,.2)
# t = t0+num.arange(0,8,0.01)

# define time interferometric acquisitions
tint =  times[2] - times[1]

# define coseismic time
tco = times[0]

# def Heaviside(t):
#         h=np.zeros((len(t)))
#         h[t>=0]=1.0
#         return h

# extract pyrocko result
N = result.request.targets[0].coords5[:, 2]/1000
E = result.request.targets[0].coords5[:, 3]/1000
result = result.results_list[0][0].result
components = result.keys()

# surface displacement matrix (BIL format)
disp = np.zeros((Ninsar+len(t)*Ngps,4))

# create fake interseismic surface displacements 
vint = 0.004
disp[:Ninsar,0] = vint*tint 

for i in xrange(Ngps):
    d = as_strided(disp[Ninsar+i*len(t):Ninsar+(i+1)*len(t),:])
    # Heaviside function define in kernel.py
    d[:,0] = vint*(t-t0)*Heaviside(t-t0) # los component
    d[:,1] = vint*(t-t0)*Heaviside(t-t0) # east component
    d[:,2] = vint*(t-t0)*Heaviside(t-t0) # down component
    d[:,3] = vint*(t-t0)*Heaviside(t-t0) # north component

# add coseismic surface displacements compute from engine
# print result['displacement.los'][:Ninsar]
# print
disp[:Ninsar,0] -= result['displacement.los'][:Ninsar] 
for i in xrange(Ngps):
    d = as_strided(disp[Ninsar+i*len(t):Ninsar+(i+1)*len(t),:])
    d[:,0] -= result['displacement.los'][Ninsar+i]*Heaviside(t-tco)
    d[:,1] -= result['displacement.e'][Ninsar+i]*Heaviside(t-tco)
    d[:,2] -= result['displacement.d'][Ninsar+i]*Heaviside(t-tco)
    d[:,3] -= result['displacement.n'][Ninsar+i]*Heaviside(t-tco)

# vpost = 0.001
# # add postseismic signal (define in kernel.py)
# post = postseismic(tini = tco, tend= tco+1., Mfunc=1)
# post = flatten(post)
# for k in xrange(len(post)):
#     disp[:Ninsar,0] -= vpost*post[k].g(times[2])
#     for i in xrange(Ngps):
#         d = as_strided(disp[Ninsar+i*len(t):Ninsar+(i+1)*len(t),:])
#         d[:,0] -= vpost*post[k].g(t) 
#         d[:,1] -= vpost*post[k].g(t)
#         d[:,2] -= vpost*post[k].g(t)
#         d[:,3] -= vpost*post[k].g(t)
#     plt.plot(t,post[k].g(t),label=post[k].name)
# plt.legend()
# plt.show()
# sys.exit()

# Synthetic sismograms



# Add random noise
print 'Add random noise to synthetic data'
sig_insar = 0.005 
sig_gps = 0.002 
print 'sigmad_insar: {}, sigmad_gps: {}'.format(sig_insar,sig_gps)
print

rseed = 231
# randow value that produce the same value for same seeds
rstate = num.random.RandomState(rseed)
xr = num.zeros((Ninsar+len(t)*Ngps, 4))
# print np.max(result['displacement.los'][:Ninsar]), np.min(result['displacement.los'][:Ninsar])
xr[:, 0] = rstate.uniform(-sig_insar, sig_insar, size=Ninsar+len(t)*Ngps) # los component
xr[:, 1] = rstate.uniform(-sig_gps, sig_gps, size=Ninsar+len(t)*Ngps) # east component
xr[:, 2] = rstate.uniform(-sig_gps, sig_gps, size=Ninsar+len(t)*Ngps) # down component
xr[:, 3] = rstate.uniform(-sig_gps, sig_gps, size=Ninsar+len(t)*Ngps) # north component
disp += xr

# fig, _ = plt.subplots(4,1,figsize=(18,6))
# # plot first GPS station surface displacements
# for i, ax, dspl in zip(np.arange(4),fig.axes,components):
#     ax.plot(t,disp[Ninsar+len(t)*0:Ninsar+len(t)*1,i])
#     ax.set_ylabel(dspl+' [m]')
#     ax.set_xlabel('Time')
#     fig.autofmt_xdate()

# plt.show()
# sys.exit()

##############################################
#            Save Foward model               #
##############################################

# #### Uncomment to create new data files ######

# # save insar stack
# fid = open('./synthetic_example/insar/int_20081008-20081114.xylos','w')
# # print np.vstack([E[:Ninsar], N[:Ninsar], disp[:Ninsar,0]]).T
# np.savetxt(fid, np.vstack([E[:Ninsar], N[:Ninsar], disp[:Ninsar,0]]).T ,header = 'x(km)     y(km)    los(m/yr)  ',comments = '# ')
# fid.write('\n')
# fid.close


# # save gps stations locations
# # fid = open('./synthetic_example/gps/synt_gps_km.txt','w')
# # print stations_name, np.vstack([stations_east, stations_north]).T
# # np.savetxt(fid, stations_name, np.vstack([stations_east, stations_north]).T ,header = ' name  x(km)  y(km) ',comments = '# ')
# # fid.write('\n')
# # fid.close

# # save gps time series
# for i in xrange(Ngps):
#     fid = open('./synthetic_example/gps/SYNT/'+stations_name[i]+'.neu','w')
#     d = as_strided(disp[Ninsar+i*len(t):Ninsar+(i+1)*len(t),:])
#     for ii in xrange(len(t)):
#         np.savetxt(fid, np.vstack([t[ii], d[ii,1], d[ii,3], d[ii,2], 0.001, 0.001, 0.005]).T)
#     fid.write('\n')
#     fid.close

# sys.exit()

#####################################################
############ OPTIMISATION PARAMETERS ################
#####################################################

print 'Start Optimization...'
print

# define paths
maindir='./synthetic_example/'
outdir=maindir+'output/'


# Define Spatio-temporal functions : kernels(time, space)
# Dictionary of available functions: coseismic(), interseismic(), postseismic()
# Each functions have seral structures as attribute
# One structure can be made of several segments with connectivity and kinematic conservation properties
kernels=[
coseismic(
    name='2008 event',
    structures=[
        segment(
            # name='zongwulong',ss=0.,ds=slip,x1=4,x2=-12.5,x3=5.,length=12.,width=5.5,strike=108,dip=53,
            # sig_ss=0.,sig_ds=0.,sig_x1=0,sig_x2=0,sig_x3=0,sig_length=0.,sig_width=0.,sig_strike=0,sig_dip=0.,
            name='zongwulong',ss=0.,ds=1,x1=4,x2=-12.5,x3=5.,length=10.,width=5.,strike=108,dip=53.,
            sig_ss=0.,sig_ds=1.,sig_x1=0,sig_x2=0,sig_x3=0,sig_length=5.,sig_width=3.,sig_strike=0,sig_dip=0.,
            prior_dist='Unif',connectivity=False,conservation=False,
            )],
    date=tco,
    sigmam=1.0,
    )
    ]

# Define Temporal functions
# Dictionary of available functions: coseismic(), interseismic(), postseismic, seasonal(), ref()
basis=[
# coseismic(name='coseismic', date=t0, m=0., sigmam=0.1),
interseismic(name='interseismic', date=t0, m=0, sigmam=0.01),
# interseismic(name='interseismic', date=t0, m=vint, sigmam=0., prior_dist='Unif'),
# postseismic(tini = tco, tend= tco+1., Mfunc=1, m=vpost, sigmam=0)
]  


# Define data sets
# Available class: insarstack, insartimeseries, gpsstack, gpstimeseries, waveforms

# Define timeseries data set: time series will be clean temporally from basis functions
timeseries=[
    gpstimeseries(
        # network='synt_gps_km_short.txt',
        # reduction='SYNT', 
        network='synt_gps_km.txt',
        reduction='SYNT', # directory where are the time series
        dim=3, # [East, North, Down]: dim=3, [East, North]: dim =2
        wdir=maindir+'gps/',
        scale=1., # scale all values
        weight=1./sig_gps, # give a weight to data set
        proj=[1.,1.,1.],
        extension='.neu',
        base=[0,0,0],
        sig_base=[0,0,0],
        ),
     ]

# Define stack data set: velcoity maps, average displacements GPS vectors, interferograms, ect...
# Cannot be clean from temporal basis functions
stacks=[
    insarstack(network='int_20081008-20081114.xylos',
            reduction='Int.',wdir=maindir+'insar/',proj=projm,
            tmin= times[1], tmax=times[2], los=None,heading=None,
            weight=1./sig_insar,scale=1.,base=[0,0,0],sig_base=[0,0,0],dist='Unif'),
    ]

# Optimisation
short_optim = False # if True: fast optimization with scipy
bayesian = True # if True: bayesian exploration with Metropolis sampling
MAP = False # if True: display maximum posteriori values using functions in Scipy's optimize
niter=1000 # number of sampling for bayesian exploration
nburn=500 # number of burned sampling  


# Define profile for plot?
# Nothing done for the moment
profile=profile(name='all',x=0,y=0,l=10000,w=1000,strike=0) 


# text files for plots in gmt format
gmtfiles=[
    gmt(name='Fault traces',wdir=maindir+'gmt/',filename='faults_km.gmt',color='black',width=2.),
       ]

# define boundaries for the plots
bounds = [left,right,bottom,top]





