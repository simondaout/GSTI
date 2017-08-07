import numpy as np
import os

from kernel import *
from model import *
from gps import *
from insar import *
from structures import *
from readgmt import *

maindir='/Users/daouts/work/californie/parkfield/'
outdir='./bid/'


kernels=[
coseismic(
    name='Parkfield 2004',
    structures=[
        segment(
            name='Parkfield segment',ss=0.5,ds=0,x1=-6.4,x2=4.95,x3=0,length=40,width=13,strike=-37.6,dip=90,
            sig_ss=0.5,sig_ds=0.,sig_x1=0,sig_x2=0,sig_x3=0,sig_length=0,sig_width=0,sig_strike=0,sig_dip=0,
            prior_dist='Unif',connectivity=False,conservation=False,
            )],
    date=2004.7445,
    inversion_type='space',
    sigmam=1.0,
    )
    ]

basis=[]

profile=profile(name='all',x=0,y=0,l=10000,w=1000,strike=0)   

store='halfspace'
store_path=['/Users/daouts/work/gf_store/']

data=[
    # insarstack(network='geo_20160113-20160206_disp-m_stree_km.xy-los',
    #         reduction='T104',wdir='/Users/daouts/work/tibet/menyuan/insar/',proj=[-0.61, -0.1447,0.629],
    #         los=None,heading=None,weight=1.,scale=1),


    gpstimeseries(network='cgps_stations_km.dat',
        reduction='cgps_coseismic_4points',
        dim=2,
        wdir=maindir+'gps/timeseries/',
        scale=1.,
        weight=1.,
        proj=[1.,1.,1.],
        extension='.neu'
        ),

     ]

     
gmtfiles=[
    gmt(name='fault traces',wdir=maindir+'gmt/',filename='ca_faults_km.xyz',color='black',width=1.),
    gmt(name='coast lines',wdir=maindir+'gmt/',filename='ca_coasts_km.xyz',color='black',width=2.),
    gmt(name='river lines',wdir=maindir+'gmt/',filename='ca_rivers_km.xyz',width=2.,color='blue'),
        ]


