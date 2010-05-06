import Distribution

if __name__=="__main__":
    p = Distribution.MixtureOfLogNormals()

    print p

    
    dat = p.sample(50000)
    
    p = Distribution.MixtureOfLogNormals()
    p.histogram(dat)
    p.estimate(dat)
    p.histogram(dat,cdf=True)
    raw_input()
