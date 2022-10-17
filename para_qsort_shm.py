from multiprocessing import shared_memory, Pool
import psutil
from time import time
import numpy as np
from itertools import repeat

NUM_CPUS = psutil.cpu_count(logical=True)
FACTOR = 4
NUM_TH = FACTOR*NUM_CPUS

def first_partition(buf_name, lr, pivot, shape, dtype, n):
    shm = shared_memory.SharedMemory(buf_name)
    print('>>first partiotion', id(shm), id(shm.buf), lr, pivot, n)
    a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    (left, right) = lr
#    print(pivot, left, right, a[left], a[right])
    i = left
    j = right
    while i <= j:
        while a[i] < pivot: i += 1
        while a[j] >= pivot: j -= 1
        if i <= j:
            print(i,j)
            t = a[j]
            a[j] = a[i]
            a[i] = t
            i += 1
            j -= 1
#    print(i)
    return (n, right - i +1, i, right, pivot) if n < NUM_CPUS//2 else (n, i - left, left, i-1, pivot)

def insert_sort(l, p=0, r=2):
    for j in range(p+1, r+1):
        key = l[j]
        i = j - 1
        while i - p >= 0 and l[i] > key:
            l[i+1] = l[i]
            i -= 1
        l[i+1] = key    

def main():
    
    print('in main', shm, shm.name, id(shm), id(snpa), id(l))
    a = shm.buf
    len_data = len(a)
    n = len_data//NUM_CPUS
    chunks = [(i, i+n-1) if i < n*(NUM_CPUS-1) else (i, len_data-1) for i in range(0, n*NUM_CPUS, n)]
    print(chunks)
    m = len_data//2
    p = [a[m-1],a[m],a[m+1]]
    insert_sort(p)
    pivot = p[1]
    num_proc = [n for n in range(0, NUM_CPUS)]


    with Pool(NUM_CPUS) as pool:
        res = pool.starmap(first_partition,  zip(repeat(shm.name), chunks, repeat(pivot), 
        repeat(l.shape), repeat(l.dtype), num_proc))
#            print(sl)
#            print(res)
#             bigs = smalls = []
#             start2 = time()
#             for n, r in enumerate(res):
# #                print('>>',n,r)
#                 if n < NUM_CPUS//2:
#                     bigs = bigs + [t for t in range(r[2],r[3]+1)]
#                 else:
#                     smalls = [t for t in range(r[3],r[2]-1,-1)] + smalls
# #            print('length', len(bigs), len(smalls))
#             range_time = time() - start2
#             num_for_swap = min(len(bigs),len(smalls))//NUM_CPUS
#             bi = [iter(bigs)]*num_for_swap
#             sm = [iter(smalls)]*num_for_swap

#             pool.starmap(swaps, zip(repeat(sl), zip(*bi), zip(*sm)))

if __name__=="__main__":   
    n = 50_000_000
    l = np.random.randint(-500, 500, size=n)
    print(id(l), l.shape, l.dtype)
    start_time = time()
    shm = shared_memory.SharedMemory(create=True, size=l.nbytes)
    snpa = np.ndarray(l.shape, dtype=l.dtype, buffer=shm.buf)
    main()
    duration = time() - start_time
    print(f"\nDuration {duration} seconds")

