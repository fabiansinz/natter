import Distributions

if __name__=="__main__":
    p = Distributions.MixtureOfGaussians({'K':10})

    print p

    
    dat = p.sample(50000)

    p = Distributions.MixtureOfGaussians({'K':5})
    p.histogram(dat,cdf=True)
    p.estimate(dat)
    p.histogram(dat,cdf=True)
    raw_input()
