from sys import argv
from time import time
import os
from multiprocessing import shared_memory, set_start_method
from concurrent.futures import ProcessPoolExecutor
import numpy as np

NUM_CPUS = os.cpu_count() or 4
INSERT_MAX = 32

def insert_sort_shm(shm_name, shape, dtype, p, r):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    for j in range(p+1, r+1):
        key = a[j]
        i = j - 1
        while i - p >= 0 and a[i] > key:
            a[i+1] = a[i]
            i -= 1
        a[i+1] = key

def seq_qsort_shm(shm_name, shape, dtype, lr):
    left, right = lr
    if left >= right:
        return True
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    if right - left < INSERT_MAX:
        insert_sort_shm(shm_name, shape, dtype, left, right)
    else:
        i = left
        j = right
        part = a[(left+right)//2]
        while i <= j:
            while a[i] < part: i += 1
            while a[j] > part: j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        while j > left and a[j] == part: j -= 1
        while i < right and a[i] == part: i += 1
        if j - left > 0:
            seq_qsort_shm(shm_name, shape, dtype, (left, j))
        if right - i > 0:
            seq_qsort_shm(shm_name, shape, dtype, (i, right))
    return True

def para_qsort(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    parts = [(0, len(a) - 1)]
    
    # Subdivide parts in the main process until we have enough parts for all CPUs
    while len(parts) < NUM_CPUS:
        parts.sort(key=lambda x: x[1] - x[0], reverse=True)
        largest_part = parts[0]
        if largest_part[1] - largest_part[0] < INSERT_MAX:
            break
        
        parts.pop(0)
        left, right = largest_part
        i = left
        j = right
        part = a[(left+right)//2]
        while i <= j:
            while a[i] < part: i += 1
            while a[j] > part: j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        while j > left and a[j] == part: j -= 1
        while i < right and a[i] == part: i += 1
        
        if j - left >= 0:
            parts.append((left, j))
        if right - i >= 0:
            parts.append((i, right))
            
    # Now sort all parts in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(NUM_CPUS) as executor:
        executor.map(seq_qsort_shm, [shm_name]*len(parts), [shape]*len(parts), [dtype]*len(parts), parts)

def is_sorted(name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=name)
    l = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    if np.all(l[:-1] <= l[1:]):
        return 'Success'
    else:
        for i in range(len(l)-1):
            if l[i] > l[i+1]:
                return ('Fail', i)
        return 'Fail'

def main():
    n = 2_000_000
    print(f"Generating random array of size {n}...")
    np_array = np.random.randint(0, 500, size=n)
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    a = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    a[:] = np_array[:]
    
    start_time = time()
    para_qsort(shm.name, np_array.shape, np_array.dtype)
    duration = time() - start_time
    print(f"\nDuration {duration:.4f} seconds")
    print(is_sorted(shm.name, np_array.shape, np_array.dtype))
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    set_start_method('spawn')
    main()