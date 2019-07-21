import numpy as np
from numba import cuda

@cuda.jit("void(float32[:], float32[:], int32, complex64[:], complex64, float32[:, :])")
def compute_newton_gpu(x, y, max_iter, p, a, out):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < y.shape[0]:
        z = x[i] + 1j * y[j]
        k = 0
        for _ in range(max_iter):
            v = 0
            for l in range(p.shape[0]):
                z_pow = 1
                for _ in range(l):
                    z_pow *= z
                v += p[l] * z_pow
            if v.real*v.real + v.imag*v.imag < .1: 
                break
            w = 0
            for l in range(1, p.shape[0]):
                z_pow = 1
                for _ in range(l - 1):
                    z_pow *= z
                w += l * p[l] * z_pow
            z = z - a * v / w
            k += 1
        out[i, j] = k / max_iter

def compute_newton(x, y, max_iter, p, a = 1):
    rows, cols = x.shape[0], y.shape[0]
    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(cols / block_dim[1] + 1))
    
    stream = cuda.stream()
    x2 = cuda.to_device(np.asarray(x, dtype = np.float32), stream = stream)
    y2 = cuda.to_device(np.asarray(y, dtype = np.float32), stream = stream)
    p2 = cuda.to_device(np.asarray(p, dtype = np.complex64), stream = stream)
    out2 = cuda.device_array((rows, cols), dtype = np.float32)
    compute_newton_gpu[grid_dim, block_dim](x2, y2, max_iter, p2, a, out2)
    out = out2.copy_to_host(stream = stream)
    
    return out
