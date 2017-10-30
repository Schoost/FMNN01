import numpy as np
import scipy.linalg as sl


def powerIter(A,tol=0.00001,maxIter=100000):
    sh = np.shape(A)
    v=np.random.rand(sh[0],1)
    for i in range(0,maxIter):
        vOld = v
        v=A@v
        v=v/(sqrt(v.T@v))
        if (sl.norm(vOld-v)<tol):
            break
    return v,i



def rayleigh(A,x):
    return x.T@A@x/(x.T@x)