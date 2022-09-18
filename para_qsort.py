from sys import argv
from random import randint
from time import time
import multiprocessing
from turtle import right
import psutil
from itertools import repeat
from seq_qsort import insert_sort, seq_qsort, is_sorted

NUM_CPUS = psutil.cpu_count(logical=False)
FACTOR = 8
NUM_TH = FACTOR*NUM_CPUS


def para_qsort(l):
    n = len(l)//NUM_CPUS
    chunks = [(i, i+n-1) if i < n*(NUM_CPUS-1) else (i, len(l)-1) for i in range(0, n*NUM_CPUS, n)]
    
    print(chunks)
    
    with multiprocessing.Pool(NUM_CPUS) as pool:
        print(pool)
        stage_one = pool.starmap(thread_first_partion, zip(repeat(l), chunks))

def thread_first_partion(a, lr):
    print(id(a))
    m = len(a)//2
    p = [a[m-1],a[m],a[m+1]]
    insert_sort(p)
    pivot = p[1]
    (left, right) = lr
#    print(pivot, left, right)
    i = left
    j = right
    while i <= j:
        while a[i] < pivot: i += 1
        while a[j] > pivot: j -= 1
        if i <= j:
#            print(i,j)
            t = a[j]
            a[j] = a[i]
            a[i] = t
            i += 1
            j -= 1
    while j > left and a[j] == pivot: j -= 1
    while i < right and a[i] == pivot: i += 1
#    print(a[left:right], i)
    return a[left:right], i

def main():
    
    try:
        if len(argv) > 1:
            print(f'Opening {argv[1]}')
            data = list(map(int, open(str(argv[1]))))
        else:
            with open("random_data.txt", "w") as f:
                f.writelines((str(randint(-5000,5000)) + '\n' for _ in range(100)))
            print("Opening random_data.txt")
            data = list(map(int, open("random_data.txt")))
    except:
        print(f"Can't open {argv[1]}")
        exit()
    print(data)
    para_qsort(data)

if __name__=="__main__":
    main()