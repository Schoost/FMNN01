import scipy.linalg as sl
import numpy as np


def qrIter(A,maxIter=10000,TOL=10e-6):
	A = sl.hessenberg(A)
	
	d = np.diag(A)
	for k in range(1,maxIter):
		mu = A[-1][-1]
		Q,R = sl.qr(A-mu*np.identity(len(A[:])))
		A=R@Q + mu*np.identity(len(A[:]))	
		if (sl.norm(np.sort(d)-np.sort(np.diag(A)))<TOL):
			break	

		d = np.diag(A)
		d1 = np.diag(A,k=1)
		ind = np.argmin(d1)
		if (abs(d1[ind])<TOL and len(A) > 2):
			A1,k1 = qrIter(A[1:ind][1:ind])
			A2,k2 = qrIter(A[ind+1:-1][ind+1:-1])

			A[1:ind][1:ind]=A1
			A[ind+1:-1][ind+1:-1]=A2
			k+=k1+k2

	return A,k
n=100

A=np.random.rand(n+1,n+1)
A=A.T+A

A,k=qrIter(A)
print(A,k)
