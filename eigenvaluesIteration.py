import numpy as np
import scipy.linalg as sl


def powerIter(A,tol=0.00001,maxIter=1000):
    sh = np.shape(A)
    v=np.random.rand(sh[0],1)
    for i in range(0,maxIter):
        vOld = v
        v=A@v
        v=v/sl.norm(v)
        if (sl.norm(vOld-v)<tol or sl.norm(vOld+v)<tol):
            break
    return v,i

def inverseIter(A,mu=1,tol=0.00001,maxIter=1000):
    sh = np.shape(A)
    v=np.random.rand(sh[0],1)
    for i in range(0,maxIter):
        vOld = v
        v=sl.solve(A-mu*np.identity(sh[0]),v)
        v=v/sl.norm(v)
        if (sl.norm(vOld-v)<tol  or sl.norm(vOld+v)<tol):
            break
    return v,i
def rayleigh(A,x):
    return x.T@A@x/(x.T@x)