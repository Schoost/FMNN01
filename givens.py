import scipy.linalg as sl
import numpy as np
from math import sqrt,cos,sin

def rotation(A,i,j):
    """
    This method applies a Given's rotation on the Matrix A where position i,j will be set to 0
    """
    i=i-1
    j=j-1
    a = A[j,j]
    b = A[i,j]
    
    if b != 0.0:
        r = sqrt(a**2+b**2)
        c=a/r
        s=-b/r
        
    else:
        c=1.0
        s=0.0
        r=a
    
    G=np.zeros(np.shape(A[0:2,:]))
    G[0,[j,i]]=[c,-s]
    G[1,[j,i]]=[s, c]
    A[[j,i],:]=G@A
    A[i,j]=0.0
    return A