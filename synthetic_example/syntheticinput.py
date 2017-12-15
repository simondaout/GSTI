#!/usr/bin/env python2.7
import numpy as num
import os
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc

# GSti dependencies
from GSTI.kernel import *
from GSTI.model import *
from GSTI.gps import *
from GSTI.insar import *
from GSTI.structures import *
from GSTI.readgmt import *
from GSTI.combisource import *
from GSTI.waveform import *
from GSTI.date2dec import *

# pyrcoko dependencies
from pyrocko.gf import LocalEngine, StaticTarget, SatelliteTarget,\
    RectangularSource,Target, ws
from pyrocko import util, pile, model, config, trace, io, pile

op = os.path


# define the Green Function store for the synthetic example
store='tibet_static'
# store='test'
# if not os.path.exists(store):
#     print 'Downloading gf store from reporisitory'
#     ws.download_gf_store(site='kinherd', store_id=store)
store_path=['./synthetic_example/gfstore/']

ref_lat,ref_lon = 37.6, 95.9 

#####################################################
############ CREATE SYNTHETIC EXAMPLE ###############
#####################################################

engine = LocalEngine(store_superdirs=store_path, default_store_id=store, use_config=True)

##############################################
#            Targets                         #
##############################################

# distance in kilometer
km = 1e3

# We define a grid for the targets.
left,right,bottom,top=-35,15,-35,15
# left,right,bottom,top=-30,30,-25,25

# Synthetic GPS points 
# stations_name = [   'XIAO']
# stations_east    = [-15104.34]
# stations_north    = [-22552.38]
stations_name = [   'XIAO',    'A01',  'A02',   'A03',   'A04',   'A05' ]
stations_east    = [-15104.34, -10000.,     0.,      0.,  10000.,  10000. ]
stations_north    = [-22552.38,  10000., 10000., -10000., -10000.,  10000. ]
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
lats = num.empty(Ninsar+Ngps)
lons = num.empty(Ninsar+Ngps)
lats.fill(ref_lat)
lons.fill(ref_lon)

satellite_target = SatelliteTarget(
    lats=lats,lons=lons,
    north_shifts = np.concatenate(np.array([rnd.uniform(bottom*km, top*km, Ninsar),stations_north])),
    east_shifts= np.concatenate(np.array([rnd.uniform(left*km, right*km, Ninsar),stations_east])),
    interpolation='nearest_neighbor',
    phi=phi,
    theta=theta)

gps_target = StaticTarget(
    lats=np.full(Ngps, ref_lat),
    lons=np.full(Ngps, ref_lon),
    north_shifts=np.array(stations_north),
    east_shifts=np.array(stations_east))


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

# 2008 event
# GCMT: lon=95.75, lat=37.51, Depth:27.2km, strike=108, dip:67, rake:106, mw=6.3
# [Elliott 2011]: Depth:16.4km, strike=99, dip:58, rake:95, length=15km, width=12km, mw=6.3
# [Feng 2015]:  Depth:15km, strike=288, dip:31, rake:90, length=17km, width=8km, mw=6.3
#               Depth:16km, strike=108, dip:53, rake:117, length=17km, width=5km, mw=6.3

# ref point
# echo 95.9 37.6 | proj +proj=utm +zone=46
# 756017.17       4165390.81
# relative position
# echo 95.75 37.51 | proj +proj=utm +zone=46 | awk '{printf("%f %f\n",($1-756017.17)/1e3,($2-4165390.81)/1e3)}'
# -12952.54    -10386.69

# # north dipping plane
# east,north=-13, -10
# d = 15.
# strike=288
# dip=31.
# rake = 90 
# l = 17.
# W=8.
# mw=6.3

# 2008 event
# # south dipping place
east08,north08=-13, -10
d08 = 12.
strike08=108
dip08=0.
rake08 = 90 
l08 = 12.
W08=12.
mw08=6.3

slip08 = mw2slip(mw08,l08,W08)
time08 = '2008-11-10 01:22:10.230'

# print ref_lat , ref_lon
# print north08*km, east08*km, d08*km, W08*km, l08*km, dip08,
# print rake08, strike08, util.str_to_time(time08), slip08
# print 

co2008 = RectangularSource(
    lon= ref_lon, lat = ref_lat,
    north_shift=north08*km, east_shift=east08*km,
    depth=d08*km, width=W08*km, length=l08*km,
    dip=dip08, rake=rake08, strike=strike08,
    time=util.str_to_time(time08), 
    slip=slip08, anchor='top')

print
print 'Synthetic model:'
print co2008

patches = [co2008];
sources = CombiSource(subsources=patches)
# The computation is performed by calling process on the engine
result_2008 = engine.process(sources, [satellite_target, gps_target])

# 2009 event
# GCMT: lon=95.76, lat=37.64, Depth:12km, strike=101, dip:60, rake:83, mw=6.3
# [Elliott 2011]: Depth:4.7km, strike=100, dip:53, rake:106, length=12.2km, width=5.4km, mw=6.3
# [Feng 2015]:  Depth:5km, strike=108, dip:53, rake:90, length=xkm, width=xkm, mw=6.3
# east09,north09=-12.5,4
# d09 = 5.
# strike09=108
# dip09=53.
# rake09 = 90 
# l09 = 12.
# W09=5.5
# mw09=6.3

# lets connect the second rupture to the first one
rake09 = 90 
l09 = 12.
strike09 = strike08
dip09 = 65.
W09 = 8.
mw09 = 6.3

# vertical distance
d09 = d08 - W09*math.sin(np.deg2rad(dip09))

# horizontal distance
yp = math.cos(np.deg2rad(dip09))*W09

# shifts 
east_shift = -math.cos(np.deg2rad(strike08))*yp 
north_shift = math.sin(np.deg2rad(strike08))*yp
east09,north09= east08+east_shift, north08+north_shift

slip09 = mw2slip(mw09,l09,W09)

time09 = '2009-08-28 01:52:12.710' 
co2009 = RectangularSource(
    lon= ref_lon, lat = ref_lat,
    north_shift=north09*km, east_shift=east09*km,
    depth=d09*km, width=W09*km, length=l09*km,
    dip=dip09, rake=rake09, strike=strike09,
    time=util.str_to_time(time09),
    slip=slip09, anchor='top')

print
print 'Synthetic model:'
print co2009

patches = [co2009];
sources = CombiSource(subsources=patches)
# The computation is performed by calling process on the engine
result_2009 = engine.process(sources, [satellite_target, gps_target])

##############################################
#       Create synthetic waveforms           #
##############################################

# 2008 event
# We load the refrence event from a event file. This source will be used to
# retrieve the expected arrival times.
# events = []
# events.extend(model.load_events(filename='./synthetic_example/waveforms/2008_event.csv'))
# event = events[0]
# origin = gf.Source(
#     lat=event.lat,
#     lon=event.lon)
# base_source = gf.MTSource.from_pyrocko_event(event)
# base_source.set_origin(origin.lat, origin.lon)

# # Next follows the loading of the stations and init of targets.
# # We use the term target for a single component of a single station
# fn_stations = './synthetic_example/waveforms/stations.txt' 
# # fn_stations = './synthetic_example/waveforms/stations_short.txt'  
# stations_list = model.load_stations(fn_stations)  # load the stations file
# # for s in stations_list:
# #     s.set_channels_by_name(*'Z'.split())
# # stations = {}
# # print stations

# # we would also iterate over the components for each station.
# targets=[]
# for station in stations_list:  # iterate over all stations
#     target = Target(
#             lat=station.lat,  # station lat.
#             lon=station.lon,   # station lon.
#             store_id=store,   # The gf-store to be used for this target,
#             # we can also employ different gf-stores for different targets.
#             interpolation='multilinear',   # interp. method between gf cells
#             quantity='displacement',   # wanted retrieved quantity
#             codes=station.nsl() + ('BHZ',))  # Station and network code

#     targets.append(target)  # append all singular targets in a list

# response = engine.process(co2008, targets)
# # And then we reform the response into traces:
# synthetic_traces_08 = response.pyrocko_traces()

# response = engine.process(co2009, targets)
# # And then we reform the response into traces:
# synthetic_traces_09 = response.pyrocko_traces()

##############################################
#   Create synthetic Geodetic time series    #
##############################################

# convert some dates to decimal time
# [Eq1, Eq2, Int1_date1, Int1_date2, Int2_date1, Int2_date2]
dates = [20080827, 20081210 ,20090708, 20091021]
times = date2dec(dates)
# print times
# sys.exit()

# define time for synthetic time series 
t0 = 2005.
# t = t0+num.arange(0,8,1.)
t = t0+num.arange(0,8,0.01)

# define time interferometric acquisitions
tint1 =  times[1] - times[0]
tint2 =  times[3] - times[2]

# define coseismic time
t08 = time2dec(time08)[0]
t09 = time2dec(time09)[0]

# surface displacement matrix (BIL format)
disp = np.zeros((2*Ninsar+len(t)*Ngps,4))

# extract pyrocko results 
N = result_2009.request.targets[0].coords5[:, 2]/1000
E = result_2009.request.targets[0].coords5[:, 3]/1000
# sys.exit()

result_2009 = result_2009.results_list[0][0].result
components = result_2009.keys()

# idem for 2008 event
result_2008 = result_2008.results_list[0][0].result

# create fake interseismic surface displacements 
vint = -0.004

# ie.e TS clean form interseismic trend
# vint = 0.0

disp[:Ninsar,0] = vint*tint1
disp[Ninsar:2*Ninsar,0] = vint*tint2 

for i in xrange(Ngps):
    d = as_strided(disp[2*Ninsar+i*len(t):2*Ninsar+(i+1)*len(t),:])
    # Heaviside function define in kernel.py
    d[:,0] = vint*(t-t0)*Heaviside(t-t0) # los component
    d[:,1] = vint*(t-t0)*Heaviside(t-t0) # east component
    d[:,2] = vint*(t-t0)*Heaviside(t-t0) # down component
    d[:,3] = vint*(t-t0)*Heaviside(t-t0) # north component

# Add coseismic surface displacements compute from engine
# for the two interferograms
disp[:Ninsar,0] += result_2008['displacement.los'][:Ninsar] 
disp[Ninsar:2*Ninsar,0] += result_2009['displacement.los'][:Ninsar] 

# for the GPS times series
for i in xrange(Ngps):
    d = as_strided(disp[2*Ninsar+i*len(t):2*Ninsar+(i+1)*len(t),:])
    d[:,0] +=  (result_2009['displacement.los'][Ninsar+i]*Heaviside(t-t09) + result_2008['displacement.los'][Ninsar+i]*Heaviside(t-t08) )
    d[:,1] +=  (result_2009['displacement.e'][Ninsar+i]*Heaviside(t-t09) + result_2008['displacement.e'][Ninsar+i]*Heaviside(t-t08) )
    d[:,2] +=  (result_2009['displacement.d'][Ninsar+i]*Heaviside(t-t09) + result_2008['displacement.d'][Ninsar+i]*Heaviside(t-t08) )
    d[:,3] +=  (result_2009['displacement.n'][Ninsar+i]*Heaviside(t-t09) + result_2008['displacement.n'][Ninsar+i]*Heaviside(t-t08) )

# print result_2008['displacement.d'][Ninsar]*Heaviside(t-t08)
# print result_2009['displacement.d'][Ninsar]*Heaviside(t-t09)
# sys.exit()

# vpost = 0.001
# # add postseismic signal (define in kernel.py)
# post = postseismic(tini = t09, tend= t09+1., Mfunc=1)
# post = flatten(post)
# for k in xrange(len(post)):
#     disp[:Ninsar,0] -= vpost*post[k].g(t09)
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

# Add random noise
print 'Add random noise to synthetic data'
# sig_insar = 0.0 
# sig_gps = 0.0 
sig_insar = 0.005 
sig_gps = 0.002 
print 'sigmad_insar: {}, sigmad_gps: {}'.format(sig_insar,sig_gps)
print

rseed = 231
# randow value that produce the same value for same seeds
rstate = num.random.RandomState(rseed)
xr = num.zeros((2*Ninsar+len(t)*Ngps, 4))
# print np.max(result['displacement.los'][:Ninsar]), np.min(result['displacement.los'][:Ninsar])
xr[:, 0] = rstate.uniform(-sig_insar, sig_insar, size=2*Ninsar+len(t)*Ngps) # los component
xr[:, 1] = rstate.uniform(-sig_gps, sig_gps, size=2*Ninsar+len(t)*Ngps) # east component
xr[:, 2] = rstate.uniform(-sig_gps, sig_gps, size=2*Ninsar+len(t)*Ngps) # down component
xr[:, 3] = rstate.uniform(-sig_gps, sig_gps, size=2*Ninsar+len(t)*Ngps) # north component
disp += xr


# Add orbital ramp to the interferograms
ramp1_a,ramp1_b,ramp1_c = -0.0008, 0.0004, 0.0
print 'Add synthetic ramp: {}*y + {}*x + {}'.format(ramp1_a,ramp1_b,ramp1_c)
ramp1 = ramp1_a*N[:Ninsar] + ramp1_b*E[:Ninsar] + ramp1_c
disp[:Ninsar,0] = disp[:Ninsar,0]+ramp1

ramp2_a,ramp2_b,ramp2_c = -0.001, 0.0, 0.0
ramp2 = ramp2_a*N[:Ninsar] + ramp2_b*E[:Ninsar] + ramp2_c
print 'Add synthetic ramp: {}*y + {}*x + {}'.format(ramp2_a,ramp2_b,ramp2_c)
disp[Ninsar:2*Ninsar,0] = disp[Ninsar:2*Ninsar,0]+ramp2
print 

plotdata = False # if True: plot synthetic data before optimisation
if plotdata:

    fig, _ = plt.subplots(2,2,figsize=(10,6))
    vranges = [(disp[:2*Ninsar,0].max(),disp[:2*Ninsar,0].min())]
    lmax = np.abs([np.min(vranges), np.max(vranges)]).max()
    levels = np.linspace(-lmax, lmax, 50)

    ax = fig.axes[0]
    cmap = ax.tricontourf(E[:Ninsar], N[:Ninsar],disp[:Ninsar,0]-ramp1,
                                    cmap='seismic', levels=levels)

    fn, fe = co2008.outline(cs='xy').T/1000
    ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
    ax.plot(fe[:2],fn[:2],linewidth=2.,color='black',alpha=0.5)

    fn, fe = co2009.outline(cs='xy').T/1000
    ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
    ax.plot(fe[:2],fn[:2],linewidth=2.,color='black',alpha=0.5)

    # ax.scatter(stations_east[0] , stations_north[0] , c = 'black', s = 40, marker = '^')
    # ax.text(stations_east[0],stations_north[0],stations_name[0],color='black',fontsize='x-small')

    ax.scatter(co2008.east_shift/1000,co2008.north_shift/1000, c='black', marker='*', s=40)
    ax.text(co2008.east_shift/1000+2,co2008.north_shift/1000,dates[0],color='black',fontsize='x-small')
    ax.scatter(co2009.east_shift/1000,co2009.north_shift/1000, c='black', marker='*', s=40)
    ax.text(co2009.east_shift/1000+2,co2009.north_shift/1000,dates[1],color='black',fontsize='x-small')

    ax.set_title('{}-{} + Noise'.format(dates[0],dates[1]))
    ax.set_aspect('equal')
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')
    fig.colorbar(cmap, ax=ax, aspect=5)

    ax = fig.axes[1]
    cmap = ax.tricontourf(E[:Ninsar], N[:Ninsar],disp[:Ninsar,0],
                                    cmap='seismic', levels=levels)

    ax.set_title('{}-{} + Noise + Ramp'.format(dates[0],dates[1]))
    ax.set_aspect('equal')
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')
    fig.colorbar(cmap, ax=ax, aspect=5)

    ax = fig.axes[2]
    cmap = ax.tricontourf(E[:Ninsar], N[:Ninsar],disp[Ninsar:2*Ninsar,0]-ramp2,
                                    cmap='seismic', levels=levels)

    fn, fe = co2008.outline(cs='xy').T/1000
    ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
    ax.plot(fe[:2],fn[:2],linewidth=2.,color='black',alpha=0.5)

    fn, fe = co2009.outline(cs='xy').T/1000
    ax.fill(fe, fn, color=(0.5, 0.5, 0.5), alpha=0.5)
    ax.plot(fe[:2],fn[:2],linewidth=2.,color='black',alpha=0.5)

    # ax.scatter(stations_east[0] , stations_north[0] , c = 'black', s = 40, marker = '^')
    # ax.text(stations_east[0],stations_north[0],stations_name[0],color='black',fontsize='x-small')

    ax.scatter(co2008.east_shift/1000,co2008.north_shift/1000, c='black', marker='*', s=40)
    ax.text(co2008.east_shift/1000+2,co2008.north_shift/1000,dates[0],color='black',fontsize='x-small')
    ax.scatter(co2009.east_shift/1000,co2009.north_shift/1000, c='black', marker='*', s=40)
    ax.text(co2009.east_shift/1000+2,co2009.north_shift/1000,dates[1],color='black',fontsize='x-small')

    ax.set_title('{}-{} + Noise '.format(dates[2],dates[3]))
    ax.set_aspect('equal')
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')
    fig.colorbar(cmap, ax=ax, aspect=5)

    ax = fig.axes[3]
    cmap = ax.tricontourf(E[:Ninsar], N[:Ninsar],disp[Ninsar:2*Ninsar,0],
                                    cmap='seismic', levels=levels)
    ax.set_title('{}-{} + Noise + Ramp'.format(dates[2],dates[3]))
    ax.set_aspect('equal')
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')

    fig.colorbar(cmap, ax=ax, aspect=5)
    fig.tight_layout()

    fig, _ = plt.subplots(4,1,figsize=(10,6))
    # plot first GPS station surface displacements
    comps = ['los','east','down','north']
    ymin, ymax = 0.1, -0.1
    for i, ax, dspl in zip(np.arange(4),fig.axes,comps):
        ax.plot(t,disp[2*Ninsar+len(t)*0:2*Ninsar+len(t)*1,i])
        ax.set_ylim([ymin,ymax])
        ax.set_ylabel(dspl+' [m]')
        ax.set_xlabel('Time')
        fig.autofmt_xdate()

    plt.title('Station {} time series'.format(stations_name[0]))
    plt.show()

    # open waveforms with Pyrocko
    # trace.snuffle(synthetic_traces)
    # sys.exit()

##############################################
#            Save Foward model               #
##############################################

# # save interferograms
savedata = True  # if True: save synthetic data

if savedata==True:

    # for tr in synthetic_traces_08:
    #     io.save(tr, './synthetic_example/waveforms/2008/'+'{}.{}.{}.{}'.format(tr.network, tr.station, tr.location,tr.channel))
    # for tr in synthetic_traces_09:
    #     io.save(tr, './synthetic_example/waveforms/2009/'+'{}.{}.{}.{}'.format(tr.network, tr.station, tr.location,tr.channel))

    fid = open('./synthetic_example/insar/int_{}-{}.xylos'.format(dates[0],dates[1]),'w')
    # print np.vstack([E[:Ninsar], N[:Ninsar], disp[:Ninsar,0]]).T
    np.savetxt(fid, np.vstack([E[:Ninsar], N[:Ninsar], disp[:Ninsar,0]]).T ,header = 'x(km)     y(km)    los(m/yr)  ',comments = '# ')
    fid.write('\n')
    fid.close

    fid = open('./synthetic_example/insar/int_{}-{}.xylos'.format(dates[2],dates[3]),'w')
    # print 
    # print np.vstack([E[:Ninsar], N[:Ninsar], disp[Ninsar:2*Ninsar,0]]).T
    np.savetxt(fid, np.vstack([E[:Ninsar], N[:Ninsar], disp[Ninsar:2*Ninsar,0]]).T ,header = 'x(km)     y(km)    los(m/yr)  ',comments = '# ')
    fid.write('\n')
    fid.close

    # save gps stations locations
    print stations_name, np.vstack([stations_east, stations_north]).T
    with open('./synthetic_example/gps/synt_gps_km.txt','w') as fid:
        for ista, sta in enumerate(stations_name):
            fid.write('%s    %.1f    %.1f\n' % (sta, float(stations_east[ista]), float(stations_north[ista])))

    # save gps time series
    for i in xrange(Ngps):
        # fid = open('./synthetic_example/gps/SYNT/'+stations_name[i]+'.neu','w')
        fn_gps = op.join('./synthetic_example/gps/SYNT-DENSE', stations_name[i] + '.neu')
        d = as_strided(disp[2*Ninsar+i*len(t):2*Ninsar+(i+1)*len(t),:])
        sigma = num.full_like(t, 0.001)
        sigma_vertical = num.full_like(t, 0.005)
        gps_data = np.vstack([t, d[:, 1], d[:, 3], d[:, 2], sigma, sigma, sigma_vertical]).T
        np.savetxt(fn_gps, gps_data)

#####################################################
############ OPTIMISATION PARAMETERS ################
#####################################################

print
print 'Start Optimization...'
print

# define paths
maindir='./synthetic_example/'
outdir=maindir+'output/'
# all data load in UTM coordinates relatively to a reference point
reference = [ref_lon,ref_lat]
# define green function store
# store = 'halfspace'
# store_path=['./synthetic_example/gfstore/']

# Define Spatio-temporal functions : kernels(time, space)
# Dictionary of available functions: coseismic(), interseismic(), postseismic()
# Each functions have seral structures as attribute
# One structure can be made of several segments with connectivity and kinematic conservation properties
kernels=[
    coseismic(
        name='2008 event',
        structures=[
            segment(
                # name='xitieshan',ss=0.,ds=slip08,east=east08,north=north08,down=d08,length=l08,width=W08,strike=strike08,dip=dip08,
                # sig_ss=0.,sig_ds=0.,sig_east=0,sig_north=0,sig_down=0,sig_length=0.,sig_width=0.,sig_strike=0,sig_dip=0.,
                name='xitieshan',
                ss=0.,
                ds=1.,
                east=east08,
                north=north08,
                down=d08,
                length=l08,
                width=W08,
                strike=strike08,
                dip=dip08,

                sig_ss=0.,
                sig_ds=1,
                sig_east=0,
                sig_north=0,
                sig_down=0,
                sig_length=0.,
                sig_width=0.,
                sig_strike=0,
                sig_dip=0.,

                prior_dist='Unif',
                connectivity=False,
                conservation=False)
                ],
    date=time08, # put here the GCMT time 
    sigmam=1.0,
    ),
    coseismic(
        name='2009 event',
        structures=[
            segment(
                # name='zongwulong',ss=0.,ds=slip09,east=east09,north=north09,down=d09,length=l09,width=W09,strike=strike09,dip=dip09,
                # if conncectivity, sig_strike, sig_down, sig_east, sig_north are automatically set to zero
                # sig_ss=0.,sig_ds=1.,sig_east=0.,sig_north=0.,sig_down=0.,sig_length=0.,sig_width=0.,sig_strike=0.,sig_dip=0.,
                # prior_dist='Unif',connectivity=False,conservation=False),
                name='zongwulong',
                ss=0.,
                ds=1.,
                east=east09,
                north=north09,
                down=d09,
                length=l09,
                width=W09,
                strike=strike09,
                dip=dip09,

                sig_ss=5.,
                sig_ds=1.,
                sig_east=1.,
                sig_north=0.,
                sig_down=1.,
                sig_length=4.,
                sig_width=0.,
                sig_strike=5.,
                sig_dip=1.,

                prior_dist='Unif',
                connectivity=False,
                conservation=False)
                ],
    date=time09, # put here the GCMT time 
    sigmam=1.0)
]

# Define Temporal functions
# Dictionary of available functions: coseismic(), interseismic(), postseismic, seasonal(), ref()
time0 = '2005-01-01 00:00:0.0' 
basis=[
# coseismic(name='coseismic', date=t08, m=0., sigmam=0.1),
# sinterseismic(name='interseismic', date=time0, m=0, sigmam=0.1),
interseismic(name='interseismic', date=time0, m=vint+0.002, sigmam=0.004, prior_dist='Unif'),
# postseismic(tini = t09, tend= t09+1., Mfunc=1, m=vpost, sigmam=0)
]  

# Define data sets
# Available class: insarstack, insartimeseries, gpsstack, gpstimeseries, waveforms

# Define timeseries data set: time series will be clean temporally from basis functions
timeseries=[
    gpstimeseries(
        network='synt_gps_km.txt',
        # reduction='SYNT', 
        # network='synt_gps_km.txt',
        reduction='SYNT-DENSE', # directory where are the time series
        dim=3, # [East, North, Down]: dim=3, [East, North]: dim =2
        wdir=maindir+'gps/',
        scale=1., # scale all values
        weight=1., # give a weight to data set
        proj=[1.,1.,1.],
        extension='.neu',
        base=[0,0,0],
        sig_base=[0,0,0],
        store_id=store),
     ]

# Define stack data set: velcoity maps, average displacements GPS vectors, interferograms, ect...
# Cannot be clean from temporal basis functions
stacks=[
    insarstack(
        network='int_{}-{}.xylos'.format(dates[0],dates[1]),
        reduction='Int.1',
        wdir=maindir+'insar/',
        proj=projm,
        tmin= times[0],
        tmax=times[1],
        los=None,
        # weight=1.,scale=1.,base=[ramp1_b, ramp1_a, ramp1_c],sig_base=[0.,0.,0.],dist='Unif'),
        weight=1./sig_insar,
        scale=1.,
        base=[0., 0., 0.],
        sig_base=[0.01,0.01,0.01],
        dist='Unif',
        store_id=store),

    insarstack(
        network='int_{}-{}.xylos'.format(dates[2],dates[3]),
        reduction='Int.2',
        wdir=maindir+'insar/',
        proj=projm,
        tmin= times[2],
        tmax=times[3],
        los=None,
        # weight=1.,scale=1.,,,dist='Unif'),
        weight=1./sig_insar,
        scale=1.,
        base=[0., 0., 0.],
        # base=[ramp2_b, ramp2_a, ramp2_c],
        sig_base=[0.01,0.01,0.01],
        # sig_base=[0.,0.,0.],
        dist='Unif',
        store_id=store),
    ]

seismo=[
    # waveforms(
    #     network='stations.txt',
    #     reduction='2008',wdir=maindir+'waveforms/',event='2008_event.csv',
    #     phase='P',filter_corner=0.055,filter_order=4,filter_type='low',
    #     misfit_norm=2,taper_fade=2.0,weight=1.,base=0,sig_base=0,extension='',dist='Unif'),
    
    # waveforms(
    #     network='stations.txt',
    #     reduction='2009',wdir=maindir+'waveforms/',
    #     event='2009_event.csv', phase='P',filter_corner=0.055,filter_order=4,filter_type='low',
    #     misfit_norm=2,taper_fade=2.0,weight=1.,base=0,sig_base=0,extension='',dist='Unif')
]

# Optimisation
short_optim = True # if True: fast optimization with scipy
bayesian = False # if True: bayesian exploration with Adaptative-Metropolis sampling
MAP = False # if True: display maximum posteriori values using functions in Scipy's optimize
niter=50000 # number of sampling for exploration
nburn=10000 # number of burned sampled 


# Define profiles for InSAR plots
profiles=[
    profile(name='1',x=-13,y=-10,l=40,w=5,strike=108)
    ] 


# text files for plots in gmt format
gmtfiles=[
    gmt(name='Fault traces',wdir=maindir+'gmt/',filename='faults_km.gmt',color='black',width=2.),
       ]

# define boundaries for the plots
bounds = [left,right,bottom,top]





