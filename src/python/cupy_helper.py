try:
    import cupy
    from cupy import cuda
except ImportError:
    pass

def create_cupy_array(ptr, size, dtype, strides, shape):
    mem = cuda.UnownedMemory(ptr, size, None)
    cupy_ptr = cuda.MemoryPointer(mem, 0)
    return cupy.ndarray(shape, dtype=dtype, strides=strides, memptr=cupy_ptr)
