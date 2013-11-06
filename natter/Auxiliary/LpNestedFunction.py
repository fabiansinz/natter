from __future__ import division
import types
from numpy import max, abs, sum, zeros, sign, Inf, log, sqrt, size, array, linspace, arctan, pi, floor, ceil,ones
from numpy.random import rand
from scipy import special
import string
from  natter.DataModule import Data
import copy
from matplotlib.patches import Ellipse
from matplotlib.pyplot import figure, show,draw
import Plotting
from Errors import SpecificationError
from Numerics import digamma

class LpNestedFunction:
    """
    Represents an LpNestedFunction. The tree structure is
    passed by the context-sensitive expression tree. The grammar for tree is

    * tree = (p_index, list)
    * list = index_into_data, list | index_into_data| tree, list| tree

    p_index is the index into the second argument p, which stores
    the p of the respective inner node of the
    tree. index_into_data is the index of a single dimension of
    the data points (you can also use slices with the standard
    python notation that in a:b is [a,a+1,...,b-1]). You can get a
    good feeling of how the expression can look like by looking at
    the default argument.
    """

    def __init__(self, tree=None, p=None):
        """
        Constructs an Lp-nested function
        :param tree: A string with a valid tree expression as described above.
        :type tree: string
        :param p:   An array containing the initial values of the p exponents at the inner nodes. It must have the appropriate dimension such that each *p_index* is covered.
        :type p: numpy.array

        """
        if tree == None:
            tree = "(0,0,(1,1,(3,2,3,4,5),6,7),8:11,11,(2,12:16,16,(4,17,18,19:22,22),23),24)"
        self.tree = parsetree(tree)
        self.n = parseNoLeaves(self.tree,(),{})
        self.l = parseNoChildren(self.tree,(),{})
        if p == None:
            self.p = extractp(self.tree)
            self.p = rand(max(self.p)+1.0)+.5
        else:
            self.p = p
        self.pdict = parseP(self.tree,(),{})
        self.ipdict = iparseP(self.tree,(),{})
        self.lb = 0*self.p
        self.ub = [Inf]*len(self.p)
        self.iByI = {}
        getLeavesFromSubtree((),self,self.iByI)


    def f(self,dat):
        """
        Computes the value of the Lp-nested funtion at the vectors in
        dat. Alternatively you can directly call the object on the
        data, i.e. use *L(dat)* instead of *L.f(dat)*.

        :param dat: Data on which the LpNestedFunction will be evaluated.
        :type dat: natter.DataModule.Data
        :returns: A Data object containing the function values
        :rtype: natter.DataModule.Data

        """
        return Data(computerec(self.tree,dat.X,self.p))

    def dfdx(self,dat):
        """
        Computes the derivative of the Lp-nested function at the data points in dat.

        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array

        """
        ret = 0.0*dat.X.copy()
        self.__dfdxRec((),dat,ret)
        return ret

    def dfdp(self,dat):
        """
        Computes the derivative of the Lp-nested function w.r.t to the p at
        the data points in dat.

        IMPORTANT: The derivative will only give correct results if
        each inner node has its own p.

        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :returns:  A numpy array containing the derivatives.
        :rtype:    numpy.array

        """

        # test whether each p has only one inner node
        for k,v in self.ipdict.items():
            if len(v) > 1:
                raise SpecificationError(\
                    "Cannot compute derivative since p[%i] is placed on more than one inner node" % (k,))

        ret = ones((len(self.p),dat.numex()))*Inf
        self.__dfdpRec((),dat,ret)
        return ret


    def __dfdpRec(self,I,dat,df):
        """
        Private function used by dfdp
        :param I: Node list
        :type I: list
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :param df: array for return value
        :type df: numpy.ndarray
        :returns: List with (function value of this node and list of children)
        :rtype: list
        """
        l = self.l[I] # get no of children
        tmp = zeros((l,dat.size(1))) # stores the function values of the children
        pI = self.p[self.pdict[I]] # p of the current node
        ip = self.pdict[I] # index of the p for the current node
        children = [] # holds multiindices of all the children of that tree
        childrenk = []
        for k in range(l):
            Ik = I + (k,)
            if self.n[Ik] == 1: # if Ik is a leaf
                i = self.i(Ik)
                tmp[k,:] = abs(dat.X[i,:].copy()) # the function value (= absolute value)
                childrenk.append([])
            else: # if Ik is an inner node itself
                tmp[k,:],tmp2 = self.__dfdpRec(Ik,dat,df) # get its function value
                children += tmp2
                childrenk.append(tmp2)
        vI = sum(tmp**pI,0)**(1.0/pI) # compute the function value of this node

        # compute the derivative for this inner node
        df[ip,:] = vI/pI * (vI**-pI * sum(tmp**pI * log(tmp),0) - log(vI))


        # multiply additional factors for all children nodes
        if self.l.has_key(I):
            for k in range(l):
                for Ic in childrenk[k]:
                    ipk = self.pdict[Ic] # which index does Ik have?
                    df[ipk,:] *= vI**(1.0-pI) * tmp[k,:]**(pI-1.0)
        if self.n[I] > 1:
            return (vI,children + [I])
        else:
            return (vI,children)




    def __dfdxRec(self,I,dat,df):
        """
        Private function used by dfdx
        :param I: Node list
        :type I: list
        :param dat: Data points at which the derivatives will be computed.
        :type dat: natter.DataModule.Data
        :param df: array for return value
        :type df: numpy.ndarray
        :returns: List with (function value of this node and list of children)
        :rtype: list
        """
        l = self.l[I] # get no of children
        tmp = zeros((l,dat.size(1))) # stores the function values of the children
        pI = self.p[self.pdict[I]] # p of the current node
        for k in range(l):
            Ik = I + (k,)
            i = self.i(Ik)
            if self.n[Ik] == 1: # if Ik is a leaf
                tmp[k,:] = abs(dat.X[i,:].copy()) # the function value (= absolute value)
                df[i,:] = sign(dat.X[i,:].copy()) # the derivative of the absolute value
            else: # if Ik is an inner node itself
                tmp[k,:] = self.__dfdxRec(Ik,dat,df) # get its function value
        f = sum(tmp**pI,0)**(1.0/pI) # compute the function value of this node
        for k in range(l):
            Ik = I + (k,)
            for i in self.iByI[Ik]:
                df[i,:] *= f**(1.0-pI)  * tmp[k,:]**(pI-1.0)
        return f

    def logSurface(self):
        """
        Computes the logarithm of the surface area of the Lp-nested unit sphere.

        :returns: The logarithm of the surface area.
        :rtype: float

        """
        ret = self.n[()]*log(2)

        for I in self.l.keys(): # for all inner nodes
            p = self.p[self.pdict[I]] # get the p
            ret -= (self.l[I]-1.0)*log(p) # log(1/p_I^(l_I-1))
            tmp = self.n[I + (0,)]
            for k in range(self.l[I]-1):
                ret += special.betaln(tmp/p,self.n[I + (k+1,)]/p)
                tmp += self.n[I + (k+1,)]
        return ret

    def dlogSurfacedp(self):
        """
        Computes the derivative of the logarithm of the surface area
        of the Lp-nested unit sphere w.r.t. to the p.

        IMPORTANT: The derivative will only be correct if every inner
        node has a different p.

        :returns: The logarithm of the surface area.
        :rtype: float

        """
        ret = zeros((len(self.p),))
        for I in self.l.keys():
            lI = self.l[I]
            ip = self.pdict[I]
            p = self.p[ip]
            ret[ip] -= (lI-1.0)/p
            tmp = self.n[I + (0,)]
            for k in range(lI-1):
                ret[ip] += digamma((tmp+self.n[I + (k+1,)])/p) *  (tmp+self.n[I + (k+1,)])/p**2
                ret[ip] -= digamma(tmp/p) * tmp/p**2.0
                ret[ip] -= digamma(self.n[I + (k+1,)]/p) * self.n[I + (k+1,)]/p**2.0

                tmp += self.n[I + (k+1,)]
        return ret

    def i(self, I):
        """
        Transforms the multiindex I into a) the coefficient index if I
        corresponds to leaf, b) a tuple representing the subtree
        corresponding to I.

        :param I: A multiindex into the tree (see [SinzEtAl2009]_).
        :type I: tuple of int
        :returns: index or multiindex
        :rtype: int or tuple of int

        """
        tmp = self.tree
        for k in I:
            tmp = tmp[k+1]
        return tmp


    def plotGraph(self,F):
        """
        Plots the tree corresponding to the Lp-nested function and
        plots the columns of the linear transform F as patches at the
        leaves. For that reason the number of rows of the linear
        transorm F must be a square of an integer.

        :param F: LinearTransform containing filters to be plotted at the leaves
        :type F: natter.Transforms.LinearTransform
        """
        ptchsz = sqrt(size(F.W,0))
        tmp = sqrt(max(array(self.l.values())))
        height = self.n[()]/tmp*1.5*ptchsz
        depth = height
        eldiam = .25*ptchsz
        deltad = depth / ( max(array([len(k) for k in self.n.keys()])) + 2.0)

        fig = figure()
        fig.clf()
        ax = fig.add_axes([0,0,1,1], xlim=(0,depth), ylim=(0,height))
        self.__plotGraphRec((),(ptchsz,.5*height),ptchsz,eldiam,0.0,height,deltad,F.W,ax,fig)
        draw()
        show()

    def __plotGraphRec(self,mind,root,ptchsz,eldiam,h0,h1,dd,W,ax,fig):
        """
        Private function used by plotGraph
        :param mind: index
        :type mind: int
        :param root: center of the ellipse
        :type root: list
        :param ptchsz: patch size
        :type ptchsz: int
        :param eldiam: diameter of the ellipse
        :type eldiam: float
        :param h0: height of branch
        :type h0: float
        :param h1: height of branch
        :type h1: float
        :param dd: distance
        :type dd: float
        :param W: array of matches
        :type W: numpy.ndarray
        :param ax: axis to plot on
        :type ax: matplotlib.pyplot.axis
        :param fig: figure to plot on
        :type fig: matplotlib.pyplot.figure
        """

        # add ellipse for this node
        el = Ellipse(root, eldiam, eldiam)
        ax.add_patch(el)

        # count leaves and innder nodes at this node
        leaves = []
        nodes = []
        for k in range(self.l[mind]):
            if self.n[mind + (k,)] == 1:
                leaves.append(self.iByI[mind + (k,)][0])
            else:
                nodes.append(mind + (k,))

        # do we have leaves? If yes, there will be an extra branch for them
        if len(leaves) > 0:
            l = 1.0 + len(nodes)
        else:
            l = float(len(nodes))

        # compute the height each branch gets
        dh = (h1-h0)/(l+1.0) # one for each node, one for all leaves

        # compute new heights
        newh = linspace(h0+dh,h1-dh,l)

        # label the current node with its p
        s = r"$p_{%d} = %.2f$" % (self.pdict[mind],self.p[self.pdict[mind]])
        angle = arctan(( (h1-h0)*.5-dh )/dd)/(2*pi)*360.0
        ax.text(root[0] - 2*eldiam, root[1] + eldiam,s,
             rotation= angle,
             horizontalalignment = 'left',
             verticalalignment   = 'bottom')

        # draw arrows to nodes and call the plotting routine recursively
        hc = 0
        for no in nodes:
            ax.arrow(root[0]+.5*eldiam, root[1], dd - eldiam, newh[hc] - root[1])
            newroot = (root[0]+dd, newh[hc])
            self.__plotGraphRec(no,newroot,ptchsz,eldiam,newh[hc]-.5*dh,newh[hc]+.5*dh,dd,W,ax,fig)
            hc += 1

        # plot the leaves
        if len(leaves) >0:
            ax.arrow(root[0]+.5*eldiam, root[1], dd - eldiam, newh[hc] - root[1])
            ny = floor(sqrt(float(len(leaves))))
            while ny*ptchsz > dh and ny > 1:
                ny = ceil(ny/2)
            nx = ceil(len(leaves)/ny)
            a = fig.add_axes(ax2fig([root[0]+dd,newh[hc] - .5*ny*ptchsz,nx*ptchsz,ny*ptchsz],ax), \
                             xlim=(0,nx*ptchsz),ylim=(0,ny*ptchsz),autoscale_on=False)
            a.axis('off')
            Plotting.plotPatches(W[:,leaves],(nx,ny),ptchsz,ax=a)



    def __call__(self,dat):
        """
        Computes the value of the Lp-nested funtion at the vectors in
        dat. Alternatively you can directly call the object on the
        data, i.e. use *L(dat)* instead of *L.f(dat)*.

        :param dat: Data on which the LpNestedFunction will be evaluated.
        :type dat: natter.DataModule.Data
        :returns: A Data object containing the function values
        :rtype: natter.DataModule.Data

        """
        return self.f(dat)

    def __getitem__(self,key):
        # introduce slices
        if type(key) == types.SliceType:
            raise TypeError('Slices not allowed in multindices!')
        if type(key) == types.IntType:
            tmp = self.tree[key+1]
        else:
            tmp = self.tree
            for k in key:
                if type(k) == types.SliceType:
                    raise TypeError('Slices not allowed in multindices!')
                tmp = tmp[k+1]

        if type(tmp) == types.IntType:
            return LpNestedFunction('(0,' + str(tmp) + ')',[1.0])
        else:
            return LpNestedFunction(str(tmp),self.p)


    def __str__(self):
        return tostrrec(self.tree,self.p)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """
        Returns a deepcopy of the LpNestedFunction object.

        :returns: A deepcopy of the LpNestedFunction objec.
        :rtype: natter.Auxiliary.LpNestedFunction
        """
        return copy.deepcopy(self)


def ax2fig(limits,ax):
    """
    Transform limits=[left, bottom, width, height] for a sub-axes
    rectangle into axes relative coordinates.
    :param limits: rectange shape in axis coordinates
    :type limits: list
    :param ax: subaxis where the rectange coordinates reside
    :type ax: matplotlib.pyplot.axis
    :returns: rectange shape in global coordinates
    :rtype: list
    """
    (xmin,xmax) = ax.get_xlim()
    (ymin,ymax) = ax.get_ylim()
    dx = xmax-xmin
    dy = ymax-ymin
    return [limits[0]/dx,limits[1]/dy,limits[2]/dx,limits[3]/dy]


def computerec(tree,X,p):
    """
    Compute the Lp-nested function recursively
    :param tree: Lp-nested tree
    :type tree: list
    :param X: values to evaluate
    :type X: numpy.ndarray
    :param p: p-values of the Lp-nested function
    :type p: numpy.ndarray
    :returns: Lp-nested function result
    :rtype: float
    """
    ret = zeros((size(X,1),))
    for k in range(1,len(tree)):
        if type(tree[k])==types.IntType:
            ret += abs(X[tree[k],:])**p[tree[0]]
        else:
            ret += computerec(tree[k],X,p)**p[tree[0]]
    return ret**(1/p[tree[0]])


def getLeavesFromSubtree(key,L,ret):
    """
    Compute a dictionary that takes multi-indices as keys and returns
    the coefficient indices at the leaves in the subtree corresponsing
    to the multi-indices.
    """
    ret[key] = ()
    for i in range(L.l[key]):
        I = key + (i,)
        if L.n[I] == 1:
            ret[I] = (L.i(I),)
        else:
            getLeavesFromSubtree(I,L,ret)
        ret[key] += ret[I]


def iparseP(tree,mind,ret):
    """
    Compute a dictionary that stores the multi-indices that have a specific p-index.
    """
    if not ret.has_key(tree[0]):
        ret[tree[0]] = []
    ret[tree[0]].append(mind)
    for k in range(1,len(tree)):
        if type(tree[k]) == types.TupleType:
            ret = iparseP(tree[k],mind + (k-1,),ret)
    return ret


def parseP(tree,mind,ret):
    """
    Compute a dictionary that stores the index of a specific p for a given multiindex.
    """
    for k in range(1,len(tree)):
        if type(tree[k]) == types.TupleType:
            ret = parseP(tree[k],mind + (k-1,),ret)
    ret[mind] = tree[0]
    return ret


def parseNoChildren(tree,mind,ret):
    """
    Returns a dictionary that stores the number of children for INNER
    nodes. The dictionary uses multi-indices as keys.
    """
    n = 0
    for k in range(1,len(tree)):
        if type(tree[k]) == types.IntType:
            n += 1
            #ret[mind + (k-1,)] = 1.0
        else:
            ret = parseNoChildren(tree[k],mind + (k-1,),ret)
            n += 1
    if n == 1:
        raise SpecificationError("Inner nodes must have at least two children!")
    ret[mind] = n
    return ret

def parseNoLeaves(tree,mind,ret):
    """
    Returns a dictionary that stores the number of leaves in the
    subtree corresponding to a certain multo-index I.
    """
    n = 0
    for k in range(1,len(tree)):
        if type(tree[k]) == types.IntType:
            n += 1
            ret[mind + (k-1,)] = 1
        else:
            ret = parseNoLeaves(tree[k],mind + (k-1,),ret)
            n += ret[mind + (k-1,)]
    ret[mind] = n
    return ret


def parsetree(s):
    """
    Function to parse the string representation of an Lp-nested tree.

    :param s: string representation of Lp-nested function
    """
    l = [elem.rstrip().lstrip() for elem in parseexpr(s.lstrip().rstrip()[1:-1])]
    ret = []
    for k in range(len(l)):
        if l[k][0] == '(':
            ret.append(parsetree(l[k]))
        elif l[k].count(':') > 0:
            tmp = [int(elem) for elem in l[k].split(':')]
            ret += range(tmp[0],tmp[1])
            # ret += range(tmp[0],tmp[1]+1)
        else:
            ret.append(int(l[k]))
    return tuple(ret)

def parseexpr(s):
    """
    Helper function to parse the string representation of an Lp-nested tree.

    :param s: string representation of Lp-nested tree
    """
    k = 0
    bc = 0
    ret = []
    lasti = 0
    while k < len(s):
        if s[k] == ',' and bc == 0:
            ret.append(s[lasti:k])
            lasti = k+1
        elif s[k] == '(':
            bc += 1
        elif s[k] == ')':
            bc -= 1
        k += 1
    if lasti <= k:
        ret.append(s[lasti:])
    return ret

def tostrrec(tree,p):
    """
    Helper function to a string representation of the Lp-nested
    function in the prompt (NOT the initialization!).
    """
    ps = "p[%d]=%.2f" % (tree[0],p[tree[0]])
    ss = []
    later = []
    for k in range(1,len(tree)):
        if type(tree[k]) == types.TupleType:
            if len(later) > 0:
                ss.append(listtostr(later))
                later = []
            tmp = tostrrec(tree[k],p)
            i = tmp.index('+')
            if k < len(tree)-1:
                ss.append(tmp.replace('\n','\n' + i*' ' + '|' + '   '))
            else:
                ss.append(tmp.replace('\n','\n' + i*' ' + '    '))
        else:
            later.append(tree[k])
    if len(later) > 0:
        ss.append(listtostr(later))
    fs = (len(ps)+1)*' ' + '+-- '
    fs2 = (len(ps)+1)*' ' + '|   '
    for k in range(1,len(ss)):
        ss[k] = fs + ss[k]
    ss[0] = ps  + ' +-- ' + ss[0]
    s = string.join(ss,'\n' + fs2 + '\n')
    return s

def listtostr(l):
    """
    Helper function to display index list with consecutive indices
    more compact.

    :param l: list with consecutive indices
    """
    list(l).sort()
    s = "[" + str(l[0])
    k= 1
    running = False
    while k < len(l):
        if l[k-1] + 1 < l[k]:
            s += ":%d,%d" % (l[k-1],l[k])
            running = False
        else:
            running = True
        k += 1
    if running:
        s += ":%d]" % (l[-1]+1,)
    else:
        s += "]"
    return s

def extractp(tree):
    """
    Extracts the different p indices of an Lp-nested function.

    :param tree: Lp-nested tree
    """
    ret = [tree[0]]
    for k in range(len(tree)):
        if type(tree[k]) == types.TupleType:
            ret += extractp(tree[k])
    return ret



