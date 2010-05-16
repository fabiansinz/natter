from __future__ import division
from numpy import zeros, std,ceil,max,min, histogram, where, dot, log, array, sum


def marginalEntropy(dat,method='JK'):
    n,m = dat.size()
    H = zeros((n,1))
    hfunc = None

    if method=='JK':
        print "\tEstimating marginal entropies with jackknifed MLE estimator"
        hfunc = marginalEntropyJK
    elif method=='MLE':
        print "\tEstimating marginal entropies with MLE estimator"
        hfunc = marginalEntropyMLE
    elif method=='CAE':
        print "\tEstimating marginal entropies with coverage adjusted MLE estimator"
        hfunc = marginalEntropyCAE
    elif method=='MM':
        print "\tEstimating marginal entropies with Miller-Madow estimator"
        hfunc = marginalEntropyMM

    for i in xrange(n):
        hn = 3.49* std(dat.X[i,:]) * m**(-1/3)
        bins = ceil((max(dat.X[i,:])-min(dat.X[i,:]))/hn)
        H[i] = hfunc(dat.X[i,:],bins)

    return H

def marginalEntropyJK(x,bins):
    n = len(x)
    N,c = histogram(x,bins=bins)
    hn = c[1]-c[0]; # get bin width
    N = N[where(N)]
  
    H = 0
    for k in xrange(len(N)):
        Ntmp = array(N)
        Ntmp[k] = Ntmp[k]-1
        Ntmp = Ntmp/(n-1)
        Ntmp = Ntmp[where(Ntmp)]
        H = H - N[k]* dot(Ntmp,log(Ntmp))
    return -  dot(N , log(1/n*N)) - (n-1)/n * H + log(hn);

def marginalEntropyMLE(x,bins):
    n = len(x)
    h,c = histogram(x,bins=bins)
    h = h/n
    hn = c[1]-c[0] # bin width
    h = h[where(h)]
    return -dot(h,log(h)) + log(hn)

def marginalEntropyCAE(x,bins):
    n = len(x)
    N,c = histogram(x,bins=bins)
    hn = c[1]-c[0] # bin width
    N = N[where(N)]
    C = 1 - len(where(N==1))/(n+1)
    N = C*N/n
  
    return -sum( N *log(N) / (1-(1-N)**n)) + log(hn)

def marginalEntropyMM(x,bins):
      n = len(x)
      N,c = histogram(x,bins=bins)
      hn = c[1]-c[0] # get bin width
      N = N[where(N)]
  
      return - 1/n * dot(N ,log(1/n*N)) + (len(N) - 1)/ 2 / n  + log(hn)
