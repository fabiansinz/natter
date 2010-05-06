import Distribution

if __name__=="__main__":
    p = Distribution.MixtureOfGaussians({'K':10})

    print p

    
    dat = p.sample(50000)

    p = Distribution.MixtureOfGaussians({'K':5})
    p.histogram(dat,cdf=True)
    p.estimate(dat)
    p.histogram(dat,cdf=True)
    raw_input()
