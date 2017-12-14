import numpy as np
import math,sys
from matplotlib import pyplot as plt
from GSTI.flatten import *
from GSTI.date2dec import *

from pyrocko import util

class pattern:
    def __init__(self,name,date,inversion_type,m,sigmam,prior_dist):
        self.name=name
        self.date=date
        self.inversion_type=inversion_type

        # initial values and uncertainties
        self.m = m
        self.sigmam=sigmam
        self.dist=prior_dist

    def gp(self,t):
        dt=0.001
        return (self.g(t)-self.g(t+dt))/dt

def Heaviside(t):
        t = np.atleast_1d(t)
        h=np.zeros((len(t)))
        h[t>=0]=1.0
        return h

def Box(t):
        return Heaviside(t+0.5)-Heaviside(t-0.5)

class coseismic(pattern):
        def __init__(self,structures=[],name='',date=0.,inversion_type='space',m=1., sigmam=0.,prior_dist='Unif'):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.t0=time2dec(date)[0]
          self.seismo = True

          # segments associated to kernel
          self.structures = structures 
          if len(self.structures)>0:
            inversion_type = 'space'
          self.Mstr = len(self.structures)
          # each structures can have several segments
          self.Mseg = sum(map((lambda x: getattr(x,'Mseg')),self.structures))
          segments = []
          segments.append(map((lambda x: getattr(x,'segments')),self.structures))
          self.segments = flatten(segments)
          # set time event for all patch
          map((lambda x: setattr(x,'time',util.str_to_time(self.date))),self.segments)

          # print self.Mstr, self.Mseg, self.segments[0].name
          # sys.exit()

        def g(self,t):
          # print self.t0
          # print t
          # print Heaviside(t-self.t0)
          # sys.exit()
          return Heaviside(t-self.t0)

def postseismic(tini,tend,Mfunc,structures=[],name='',inversion_type='time',m=1.,sigmam=0.,prior_dist='Unif'):

          self.seimo = False
          # segments associated to kernel
          if len(structures)>0:
            inversion_type = 'space'

          tini = time2dec(tini)[0]
          tend = time2dec(tend)[0]
          date = tini
          # print tini,tend
          postseismics=[]
          T=2*(tend-tini)/(Mfunc)
          # print T
          tl=tini+(np.array(xrange(Mfunc))+1)*T/2
          # print tl
          
          postseismics.append(transienti('initial transient',structures,tini,date,inversion_type,T,m,sigmam,prior_dist))
          for j in xrange(len(tl)-1): 
              postseismics.append(transientm('transient',structures,tl[j],date,inversion_type,T,m,sigmam,prior_dist)) 
          postseismics.append(transientf('final transient',structures,tend,date,inversion_type,T,m,sigmam,prior_dist))   

          return postseismics   

class transientm(pattern):
        def __init__(self,name,structures,datedec,date,inversion_type,T,m,sigmam,prior_dist):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)
          

          self.t0=datedec
          self.T

          Mstr = len(structures)
          # each structures can have several segments
          Mseg = sum(map((lambda x: getattr(x,'Mseg')),structures))
          segments = []
          segments.append(map((lambda x: getattr(x,'segments')),structures))
          segments = flatten(segments)
          map((lambda x: setattr(x,'time',util.str_to_time(self.date))),self.segments)


        def g(self,tp):
          
          t=(tp-self.t0)/self.T
          return ((2*(t-np.sign(t)*(t**2)+0.25))*Box(t)+Heaviside(t-0.5))

class transienti(pattern):
        def __init__(self,name,structures,datedec,date,inversion_type,T,m,sigmam,prior_dist):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.t0=datedec
          self.T=T

          Mstr = len(structures)
          # each structures can have several segments
          Mseg = sum(map((lambda x: getattr(x,'Mseg')),structures))
          segments = []
          segments.append(map((lambda x: getattr(x,'segments')),structures))
          segments = flatten(segments)
          map((lambda x: setattr(x,'time',util.str_to_time(self.date))),self.segments)


        def g(self,tp):
          t=(tp-self.t0)/self.T

          return (4*(t-t**2))*Box(2*t-0.5)+Heaviside(t-0.5)

class transientf(pattern):
        def __init__(self,name,structures,date,datedec,inversion_type,T,m,sigmam,prior_dist):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.t0=datedec
          self.T=T

          Mstr = len(structures)
          # each structures can have several segments
          Mseg = sum(map((lambda x: getattr(x,'Mseg')),structures))
          segments = []
          segments.append(map((lambda x: getattr(x,'segments')),structures))
          segments = flatten(segments)
          map((lambda x: setattr(x,'time',util.str_to_time(self.date))),self.segments)

        
        def g(self,tp):
          t=(tp-self.t0)/self.T

          return (4*(t+t**2)+1)*Box(2*t+0.5)+Heaviside(t)

class interseismic(pattern):
        def __init__(self,name,structures=[],date=0.,inversion_type='time',m=1.,sigmam=0.,prior_dist='Unif'):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.seimo = False
          self.t0=time2dec(date)[0]

          # segments associated to kernel
          self.structures = structures 
          if len(self.structures)>0:
            inversion_type = 'space'

          self.Mstr = len(self.structures)
          # each structures can have several segments
          self.Mseg = sum(map((lambda x: getattr(x,'Mseg')),self.structures))
          segments = []
          segments.append(map((lambda x: getattr(x,'segments')),self.structures))
          self.segments = flatten(segments)
          map((lambda x: setattr(x,'time',util.str_to_time(date))),self.segments)

        def g(self,t):
          return (t-self.t0)*Heaviside(t-self.t0)

class reference(pattern):
        def __init__(self,name='temporal ref.',date=0.,inversion_type='time',m=0,sigmam=0.,prior_dist='Unif'):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.seimo = False
          self.Mstruc=0
          self.Mseg=0

        def g(self,t):
          return np.ones((len(t)))

class seasonalvar(pattern):
        def __init__(self,name,date=0.,inversion_type='time',m=0,sigmam=0.,prior_dist='Unif'):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.seimo = False
          self.Mstruc=0
          self.Mseg=0

        def g(self,t):
          func=np.zeros(len(t))
          for i in xrange(len(t)):
            func[i]=math.cos(2*math.pi*t[i])

          return func

class annualvar(pattern):
        def __init__(self,name,date=0.,inversion_type='time',m=0,sigmam=0.,prior_dist='Unif'):
          pattern.__init__(self,name,date,inversion_type,m,sigmam,prior_dist)

          self.seimo = False
          self.Mstruc=0
          self.Mseg=0

        def g(self,t):
          func=np.zeros(len(t))
          for i in xrange(len(t)):
            func[i]=math.cos(4*math.pi*t[i])

          return func

