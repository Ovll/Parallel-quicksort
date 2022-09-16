from random import randint
from time import time

INSERT_MAX = 32

def seq_qsort(a, left, right):
    """ sequential quicksort (array, left, right)"""
    #  sort a[left..right],try skipping equal elements in the middle
    if right - left < INSERT_MAX:
        insert_sort(a, left, right)
    else:
        i = left
        j = right
        part = a[(left+right)//2]
        while i <= j:
            while a[i] < part: i += 1
            while a[j] > part: j -= 1
            if i <= j:
                t = a[j]
                a[j] = a[i]
                a[i] = t
                i += 1
                j -= 1
        while j > left and a[j] == part: j -= 1
        while i < right and a[i] == part: i += 1
        if j - left > 0:
            seq_qsort(a, left, j)
        if right - i > 0:
            seq_qsort(a, i, right)


def insert_sort(l, p=0, r=2):
    for j in range(p+1, r+1):
        key = l[j]
        i = j - 1
        while i - p >= 0 and l[i] > key:
            l[i+1] = l[i]
            i -= 1
        l[i+1] = key
    
def is_sorted(l):
    for i in range(len(l)-2):
        if l[i]>l[i+1]: 
            return ('Fail', i)
    return ('Success')

def main():
    l = [randint(-5000,5000) for _ in range(2_000_000)]
    lst = l[:]
    start_time = time()
    seq_qsort(lst, 0, len(lst)-1)
    duration = time() - start_time
    print(is_sorted(lst))    
    
    start_time = time()
    list1 = sorted(l)
    sorted_f = time() - start_time
    print(is_sorted(list1))
    
        
    print(f"Duration {duration} seconds")
    print(f'Built-in sorted {sorted_f} seconds')

if __name__=='__main__':
    main()