#!/usr/bin/env python2.7

print 
print '# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'
print '#                                                                   #'
print '#         Nonlinear 3-dimentional inversion for fault geometry      #'
print '#         and slip through the entire seismic cycle                 #'
print '#                                                                   #'
print '# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #'


import numpy as np
#from numpy.lib.stride_tricks import as_strided
import math

from os import path, environ
from sys import argv,exit,stdin,stdout

import getopt
import time

import scipy.optimize as opt
from pylab import *

#import multiprocessing
import pymc

def usage():
    print ''
    print 'optimize.py : Nonlinear 3-dimentional inversion of fault geometry and associated'
    print 'slip rates throught the entire seismic cycle'
    print 'usage: optmize.py input_file'
    print ''


print
print "---------------------------------------------------------------------------"
print 'Read input file'
print "---------------------------------------------------------------------------"
print

try:
    opts,args = getopt.getopt(sys.argv[1:], "h", ["help"])
except getopt.GetoptError, err:
    print str(err)
    print "for help use --help"
    sys.exit()

for o in opts:
    if o in ("-h","--help"):
       usage()
       sys.exit()
    else:
       assert False, "unhandled option"
       sys.exit()

if 1 == len(argv):
  usage()
  assert False, "no input file"

elif 2 == len(argv):
  fname = sys.argv[1]
  print
  print 'input file :', fname
  sys.path.append(path.dirname(path.abspath(fname)))
  exec ("from " + path.basename(fname)+ " import *")
else:
  assert False, "too many arguments"

# Create directories for output files
inv.outdir = outdir
if not os.path.exists(inv.outdir):
        os.makedirs(inv.outdir)
outgps = inv.outdir+'gps/'
if not os.path.exists(outgps):
        os.makedirs(outgps)
outinsar = inv.outdir+'insar/'
if not os.path.exists(outinsar):
        os.makedirs(outinsar)
outmap = inv.outdir+'map/'
if not os.path.exists(outmap):
        os.makedirs(outmap)
outpro = inv.outdir+'profiles/'
if not os.path.exists(outpro):
        os.makedirs(outpro)
outstat = inv.outdir+'stat/'
if not os.path.exists(outstat):
        os.makedirs(outstat)


inv = inversion(
kernels=kernels,
basis=basis,
timeseries=timeseries,
stacks=stacks,
seismo=seismo,
profiles=profiles,
store_path=store_path,
store=store,
gmtfiles=gmtfiles,
bounds=bounds,
ref=reference,
  )

# build data matrix
inv.d = inv.build_data()
# sys.exit()

#load kernels and build priors
print inv.info()
for i in xrange(inv.Mker):
    for j in xrange(inv.kernels[i].Mseg):
        inv.kernels[i].segments[j].build_prior()
        print inv.kernels[i].segments[j].info()

# build prior model
m_init = inv.build_prior()

### TESTING ###
inv.build_gm()
inv.residual()
# sys.exit()

# plots
# inv.plot_ts_GPS()
# inv.plot_InSAR_maps()
# plt.show()
# # inv.plot_waveforms()
# sys.exit()

print
print "---------------------------------------------------------------------------"
print ' Optimization'
print "---------------------------------------------------------------------------"
print

print 'Lenght of the data vector:', len(inv.d)
print 'Number of Free parameters:', len(inv.priors)
print

print 'Optmized parameters:'
bnd=column_stack((inv.mmin,inv.mmax))
for i in xrange(len(bnd)): 
  print 'bounds for parameter {}: {}'.format(inv.sampled[i],bnd[i])
print

if short_optim:

  t = time.time() 
  # res = opt.minimize(inv.residualscalar,inv.priors,method='SLSQP',bounds=bnd)
  # res = opt.minimize(inv.residualscalar,inv.priors,method='L-BFGS-B',bounds=bnd)
  # res = opt.fmin_slsqp(inv.residualscalar,inv.priors,bounds=bnd)
  res = opt.differential_evolution(inv.residualscalar, bounds=bnd,maxiter=niter,polish=False,disp=True)
  
  elapsed = time.time() - t
  print
  print "Time elapsed:", elapsed
  print
  print res


if bayesian:

  stochastic = pymc.Normal('Data', 
  mu = pymc.Deterministic(eval = inv.foward,
                  name = 'Foward model',
                  parents = {'theta': inv.priors},
                  doc = 'Deterministic function',
                  verbose = 0,
                  plot=True),
  tau = inv.Cov(), 
  value = inv.build_data(), 
  observed = True,
  ) 

  # Parameters = locals()
  Parameters = pymc.Model(inv.priors)
  model = pymc.MCMC(Parameters)

  if MAP:
    map_ = pymc.MAP(Parameters)
    map_.fit()

  t = time.time()
  for p in inv.priors:
    # model.use_step_method(pymc.Metropolis, p)
    model.use_step_method(pymc.AdaptiveMetropolis, p)
  
  model.sample(iter = niter, burn = nburn, thin=1)
  elapsed = time.time() - t
  print
  print "Time elapsed:", elapsed
  print


  # with model:

    # foward = inv.foward(inv.priors)

    # stochastic = pymc.Normal(
    #   'Data', 
    #   mu = foward,
    #   sd = inv.Cov(), 
    #   observed = inv.build_data(),
    #   ) 

    # if MAP:
    #   map_estimate = pymc.find_MAP(model=model)
    #   print map_estimates
    #   sys.exit()


    # t = time.time()
    # trace = pm.sample(iter = 1000, burn = 500,progressbar=True)
    # elapsed = time.time() - t
    # print
    # print "Time elapsed:", elapsed
    # print
    # pymc.summary(trace)
    # _ = pymc.traceplot(trace,format = 'eps')


print
print "---------------------------------------------------------------------------"
print ' Posterior model'
print "---------------------------------------------------------------------------"
print

# compute residual for plots
inv.residual()

# print results
for i in xrange(inv.Nmanif):
    print 'Network:', inv.manifolds[i].network
    inv.manifolds[i].printbase()
print

for i in xrange(inv.Mker):
    for j in xrange(inv.kernels[i].Mseg):
        print inv.kernels[i].segments[j].info()

# plot results 
inv.plot_ts_GPS()
inv.plot_InSAR_maps()
if bayesian:
  for i in xrange(len(inv.faults)):
    pymc.Matplot.plot(model.trace(inv.faults[i][:]),format = 'eps',path = outstat)
plt.show()
inv.plot_waveforms()








            






