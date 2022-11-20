from sys import argv
from time import time
from psutil import cpu_count
from itertools import repeat
from multiprocessing import shared_memory, set_start_method
from concurrent.futures import ProcessPoolExecutor
import sys
import os.path
import numpy as np


NUM_CPUS = cpu_count(logical=True)
INSERT_MAX = 32

def first_partition(name, shape, dtype, lr, pivot, n):    
    existing_shm = shared_memory.SharedMemory(name=name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    (left, right) = lr
    i = left
    j = right
    while i < j:
        while a[i] < pivot and i < j:
            i += 1

        while a[j] >= pivot and i < j:
            j -= 1

        if i < j:
            a[i],a[j] = a[j],a[i]

    return ((i, right, 0 if i==right and a[right]<pivot else right-i+1, True) 
        if n < NUM_CPUS//2 
        else 
        (left, right if i==right and a[right]<pivot else i-1, 
        right-left+1 if i==right and a[right]<pivot else i-left, False))

def partition(shm_name, shape, dtype, lr):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    left, right = lr
    n = (right-left+1)//NUM_CPUS
    chunks = [(i, i+n-1) if i < (right-n-NUM_CPUS) else (i, right) 
                for i in range(left, right-NUM_CPUS, n)]
    m = (left+right)//2 - 1
    insert_sort(shm_name, shape, dtype, m-3, m+3)
    pivot = a[m]
    num_proc = [n for n in range(NUM_CPUS)]

    with ProcessPoolExecutor(NUM_CPUS) as executor:
        res = executor.map(first_partition, repeat(shm_name), repeat(shape), 
                           repeat(dtype), chunks, repeat(pivot), num_proc)
    rl = [list(r) for r in res]                 # res - generator
    # Creating table for swaps, it needs to be done not in parallel.
    rls = []
    p = chunk = 0
    q = NUM_CPUS-1
    while p<q:
        if p >= NUM_CPUS//2:
            num = chunks[p][1]-chunks[p][0]+1
            if rl[p][2] == 0:
                rl[p][3] = True
                rl[p][0], rl[p][1] = chunks[p]
                rl[p][2] = num  # All elements in chunk available for swap.
            elif rl[p][2] < num:
                rl[p][3] = True
                rl[p][2] = num-rl[p][2]
                rl[p][0] = rl[p][1]+1
                rl[p][1] = chunks[q][1]
            else:
                p += 1
                continue
        if q < NUM_CPUS//2:
            num = chunks[q][1]-chunks[q][0]+1
            if rl[q][2] == 0:
                rl[q][3] = False
                rl[q][0], rl[q][1] = chunks[q]
                rl[q][2] = num
            elif rl[q][2] < num:
                rl[q][3] = False
                rl[q][2] = num-rl[q][2]
                rl[q][1] = rl[q][0] - 1
                rl[q][0] = chunks[q][0]
            else:
                q -= 1
                continue
        if rl[p][2] == 0:
            p += 1
            continue
        if rl[q][2] == 0:
            q -= 1
            continue
        cond = rl[p][2] - rl[q][2]
        chunk = min(rl[p][2], rl[q][2])
        if cond > 0:
            rls.append((rl[p][0], rl[q][1], chunk,))
            rl[p][0] += chunk
            rl[p][2] -= chunk
            rl[q][2] -= chunk
            rl[q][1] -= chunk
            q -= 1
        elif cond < 0:
            rls.append((rl[p][0], rl[q][1], chunk,))
            rl[p][0] += chunk
            rl[p][2] -= chunk
            rl[q][1] -= chunk
            rl[q][2] -= chunk
            p+=1
        else:
            rls.append((rl[p][0], rl[q][1], chunk,))
            rl[p][0] += chunk
            rl[p][2] -= chunk
            rl[q][1] -= chunk
            rl[q][2] -= chunk
            q -= 1
            p += 1

    with ProcessPoolExecutor(NUM_CPUS) as executor:
        executor.map(swaps, repeat(shm_name), repeat(shape), repeat(dtype), 
                     rls)

    if rls != []:
        cch = 0
        for i, ch in enumerate(chunks):
            if ch[0] < rls[-1][0] < ch[1]:
                cch = i
                break
        for x in range(cch, NUM_CPUS):
            if rl[x][3]:
                if rl[x][2]:
                    parts = [(left, rl[x][0]-1), (rl[x][0], right)]
                    break
            else:
                if rl[x][2] == 0:
                    parts = [(left, chunks[x][0]-1), (chunks[x][0], right)]
                    break
                elif rl[x][2] < chunks[x][1]-chunks[x][0]+1:
                    parts = [(left, rl[x][1]), (rl[x][1]+1, right)]
                    break
    else:
        parts = [(left, right)]
    return parts

def para_qsort(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    lr = (0, len(a)-1)
    parts = partition(shm_name, shape, dtype, lr)
    with ProcessPoolExecutor(NUM_CPUS) as executor:
        while len(parts) < NUM_CPUS:
            res = list(executor.map(partition, repeat(shm_name), 
                       repeat(shape), repeat(dtype), parts))
            parts = [item for r in res for item in r]
    with ProcessPoolExecutor(NUM_CPUS) as executor:
        executor.map(seq_qsort, repeat(shm_name), repeat(shape), 
                     repeat(dtype), parts)

def swaps(name, shape, dtype, rls_item):
    existing_shm = shared_memory.SharedMemory(name=name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    i, j, chunk = rls_item
    for ch in range(chunk):
        a[i+ch], a[j-ch] = a[j-ch], a[i+ch]
    return True

def seq_qsort(shm_name, shape, dtype, lr):
    """Sequential quicksort (shm_name, shape, dtype, (left, right))"""
    #  Sort a[left..right],try skipping equal elements in the middle
    left, right = lr
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    if right - left < INSERT_MAX:
    #  Using insertion sort for small sized array
        insert_sort(shm_name, shape, dtype, left, right)
    else:
        i = left
        j = right
        part = a[(left+right)//2]
        while i <= j:
            while a[i] < part: i += 1
            while a[j] > part: j -= 1
            if i <= j:
                a[i],a[j] = a[j],a[i]
                i += 1
                j -= 1
        while j > left and a[j] == part: j -= 1
        while i < right and a[i] == part: i += 1
        if j - left > 0:
            seq_qsort(shm_name, shape, dtype, (left, j))
        if right - i > 0:
            seq_qsort(shm_name, shape, dtype, (i, right))
    return True

def insert_sort(name, shape, dtype, p, r):
    existing_shm = shared_memory.SharedMemory(name=name)
    a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    for j in range(p+1, r+1):
        key = a[j]
        i = j - 1
        while i - p >= 0 and a[i] > key:
            a[i+1] = a[i]
            i -= 1
        a[i+1] = key
    
def is_sorted(name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=name)
    l = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    i = 0
    while i < len(l)-1:
        if l[i] > l[i+1]: 
            return ('Fail', i)
        i+=1
    return ('Success')

def read_from_file():
    try:
        if len(argv) > 1:
            print(f'Opening {argv[1]}')
#            data = np.fromfile(str(argv[1]), dtype=int, sep='\n')
        else:
#            data = np.fromfile("random_data.txt", dtype=int, sep='\n')
            print("Opening random_data.txt")
            data = list(map(int, open("random_data.txt")))
    except:
        print(f"Can't open {argv[1]}")
        exit()

def main():
    n = 20_000_000
    np_array = np.random.randint(0, 500, size=n)
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    a = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    a[:] = np_array[:]
    start_time = time()
    para_qsort(shm.name, np_array.shape, np_array.dtype)
#    seq_qsort(shm.name, np_array.shape, np_array.dtype, (0, n-1))
#    insert_sort(shm.name, np_array.shape, np_array.dtype, 0, 6)
    duration = time() - start_time
    print(f"\nDuration {duration} seconds")
    print(is_sorted(shm.name, np_array.shape, np_array.dtype))

if __name__=="__main__":
    set_start_method('spawn')
    main()
