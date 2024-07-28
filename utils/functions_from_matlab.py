import numpy as np
from time import time

def element(A, i=None, j=None):
    # Handling default arguments
    if i is None:
        i = slice(None)  # Selecting all rows
    if j is None:
        j = slice(None)  # Selecting all columns
    if isinstance(A, np.ndarray):  # Matrix
        if min(A.shape) > 1:  # Check if A is not a vector
            if isinstance(i, list) and isinstance(j, list):
                e = A[np.ix_(i, j)]
            else:
                e = A[i, j]
        else:
            e = A[i] if isinstance(i, list) else A[i]
    return e


# Performs Gram-Schmidt Method
def krylov_ata(A, v1=None, k=10, full=1, reortho=2,counter = 0):
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    if v1 is None:
        v1 = np.random.randn(A.shape[1])
    k = min(k, min(A.shape))
    alpha = np.zeros(k)
    beta = np.zeros(k if full else k-1)
    len_v1 = len(v1)
    if reortho:
        V = np.zeros((len_v1, k + 1))
        V[:, 0] = v1 / np.linalg.norm(v1)
        U = np.zeros((A.shape[0], k))
    else:
        v = v1 / np.linalg.norm(v1) 
    for j in range(k):
        ## b2 = V[:, j].copy()
        start = time()   
        if reortho:
            r = A @ V[:, j]
            ## r = A.dot(b2)
            # if j == 0 and reortho == 2:
            #    U = np.zeros((len(r), k))
        else:
            r = A @ v
        t1 += time() - start
        counter += 1
        start = time()
        if j > 0:
            if reortho == 2:
                r -= beta[j-1] * U[:, j-1]
                r -= U[:, :j] @ (U[:, :j].T @ r)
            # else: ####CHECK WHAT TO DO HERE
            #    r -= beta[j-1] * u
        alpha[j] = np.linalg.norm(r)
        if alpha[j] == 0:
            break
        t2 += time() - start
        start = time()
        if reortho == 2:
            U[:, j] = r / alpha[j]
            r = A.T @ U[:, j]
            counter += 1
        else:
            u = r / alpha[j]
            r = A.T @ u
            counter += 1
        t3 += time() - start
        start = time()
        if reortho:
            r -= alpha[j] * V[:, j]
            r -= V[:, :j+1] @ (V[:, :j+1].T @ r)
        else:
            r -= alpha[j] * v
        if j < k - 1 or full:
            beta[j] = np.linalg.norm(r)
            if beta[j] == 0:
                break

            if reortho:
                V[:, j+1] = r / beta[j]
            else:
                v = r / beta[j]
        t4 += time() - start
    if not reortho:
        V = v
    if reortho < 2:
        U = u
    print('ATA')
    print(t1)
    print(t2)
    print(t3)
    print(t4)
    return V, U, alpha, beta, counter

# Expands the number of basis vectors in the space
def krylov_ata_expand(A, V, U, c, k=10, counter = 0):
    m = V.shape[1]
    V = np.concatenate((V, np.zeros((V.shape[0], k))), axis=1)
    U = np.concatenate((U, np.zeros((U.shape[0], k))), axis=1)
    alpha = np.zeros(k)
    beta = np.zeros(k)
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for j in range(m, k + m):
        start = time()
        if j == m:
            r = A @ V[:, j - 1] - (U[:, :j - 1] @ c.T)
            counter += 1
        else:
            r = A @ V[:, j - 1] - beta[j - m - 1] * U[:, j - 2]
            counter += 1
        t1 += time() - start
        start = time()
        r -= - U[:, :j - 1] @ (U[:, :j - 1].T @ r)
        alpha[j - m] = np.linalg.norm(r)
        if alpha[j - m] == 0:
            break
        U[:, j - 1] = r / alpha[j - m]
        t2 += time() - start
        start = time()
        r = A.T @ U[:, j - 1] - alpha[j - m] * V[:, j - 1]
        t3 += time() - start
        start = time()
        counter += 1
        r -= V[:, :j] @ (V[:, :j].T @ r)
        beta[j - m] = np.linalg.norm(r)
        if beta[j - m] == 0:
            break
        V[:, j] = r / beta[j - m]
        t4 += time() - start
    print('ATAE')
    print(t1)
    print(t2)
    print(t3)
    print(t4)
    return V, U, alpha, beta, counter


def krylov_schur_svd(A, v1=None, nr=1, tol=1e-6, absrel='rel', mindim=10, maxdim=20, maxit=1000, info=1, counter = 0):
    if v1 is None:
        v1 = np.random.rand(A.shape[1])
    if mindim < nr:
        mindim = nr
    if maxdim < 2 * mindim:
        maxdim = 2 * mindim
    if absrel == 'rel':
        tol *= np.linalg.norm(A, 1)
    B = np.zeros((maxdim, maxdim + 1))
    # Slow Here
    V, U, alpha, beta, counter = krylov_ata(A, v1, mindim, counter = counter)
    #print(end1)
    #print((1,counter))
    # Bidiagonal Form for the first mindim rows and cols
    B[:mindim + 1, :mindim + 1] = np.diag(np.append(alpha, [0])) + np.diag(beta, 1)
    hist = np.zeros(maxit, dtype=np.float64)    
    np.set_printoptions(precision=15) 
    # Modified MATLAB code ordering
    # Slow Here
    # v, u, a, b , counter = krylov_ata_expand(A, V, U, B[:mindim, mindim], maxdim - mindim, counter)
    #print((2,counter))
    for k in range(maxit):
        V, U, alpha, beta, counter = krylov_ata_expand(A, V, U, B[:mindim, mindim], maxdim - mindim, counter)
        #end2 = time() - start2
        #print(end2)
        #print(alpha)
        #print(beta)
        #V, U, alpha, beta = v.copy(), u.copy(), a.copy(), b.copy()
        B[mindim: maxdim, mindim: maxdim] = np.diag(alpha) + np.diag(beta[:maxdim - mindim - 1], 1)        
        B[maxdim - 1, maxdim] = beta[maxdim - mindim - 1]
        X, sigma, Y = np.linalg.svd(B[:maxdim, :maxdim])
        # Restart of Lanczos algorithm
        V = np.concatenate((element(V[:, :maxdim] @ Y, list(range(V.shape[0])), list(range(mindim))), V[:, maxdim:maxdim + 1]), axis=1)
        U = element(U[:, :maxdim] @ X, list(range(U.shape[0])), list(range(mindim)))   
        c = B[:, maxdim]
        e = (c @ X)[:mindim]
        B[:mindim, :mindim + 1] = np.concatenate((np.diag(sigma[:mindim]), e.reshape(-1, 1)), axis=1)
        err = np.linalg.norm(e[:nr])
        hist[k] = err 
        if info:
            print(str(k) + ": " + str(hist[k]))
        if err < tol:
            sigma = sigma[:nr]
            V = V[:, :nr]
            U = U[:, :nr]
            mvs = np.arange(1, k + 2) * (maxdim - mindim) + mindim
            print(counter)
            print(f"Found after {k + 1} iteration(s) with residual = {err}")
            return sigma, V, U, hist[:k+1], mvs, counter  
    print(counter)
    mvs = 2 * (np.arange(1, k + 2) * (maxdim - mindim) + mindim)
    if info:
        print(f"Quit after max {k + 1} iterations with residual = {err}")
    sigma = sigma[:mindim]
    V = V[:, :mindim]
    return sigma, V, U, hist, mvs, counter