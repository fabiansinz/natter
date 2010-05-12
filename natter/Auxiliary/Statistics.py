from numpy import squeeze, where, array, max


def quantile(x,p):
    x = squeeze(x)
    N = len(x)
    x.sort()

    
    mp = (array(range(N))+0.5)/N

    if p <= 0.5/N:
        return x[0]
    elif p >= (N-0.5)/N:
        return x[-1]
    else:
        I = where(mp < p)
        li = max(I[0])
        return x[li]+ ((p-mp[li])/(mp[li+1]-mp[li]))*(x[li+1]-x[li]);

