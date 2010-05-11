import Distributions

if __name__=="__main__":
    p = Distributions.MixtureOfLogNormals()

    print p

    
    dat = p.sample(50000)
    
    p = Distributions.MixtureOfLogNormals()
    p.histogram(dat)
    p.estimate(dat)
    p.histogram(dat,cdf=True)
    raw_input()
