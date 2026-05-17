using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

Console.WriteLine("Generating 200,000,000 random integers (~800 MB)...");
var rand = new Random(42);
int[] array = new int[200_000_000];
for (int i = 0; i < array.Length; i++)
{
    array[i] = rand.Next(-500, 500);
}

Console.WriteLine("Starting Multi-Threaded Parallel Vector256 QuickSort...");
var sw = Stopwatch.StartNew();

ParallelUltraSort.QuickSort(array);

sw.Stop();
Console.WriteLine($"\nFinished sorting in: {sw.Elapsed.TotalSeconds:F4} seconds");

Console.WriteLine("Validating sorted order...");
bool isSorted = true;
for (int i = 0; i < array.Length - 1; i++)
{
    if (array[i] > array[i + 1])
    {
        Console.WriteLine($"Validation Failed at index {i}: {array[i]} > {array[i + 1]}");
        isSorted = false;
        break;
    }
}

if (isSorted)
{
    Console.WriteLine("Validation Success! 200M items are perfectly sorted in parallel.");
}

public static class ParallelUltraSort
{
    private const int InsertMax = 32;
    private const int ParallelThreshold = 4_000_000;

    public static unsafe void QuickSort(int[] array)
    {
        if (array.Length > 1)
        {
            // Pin the array in memory so the garbage collector won't move it
            // while background threads are actively reading and writing to it.
            fixed (int* pArray = array)
            {
                QuickSortInternal(pArray, 0, array.Length - 1);
            }
        }
    }

    // Changed to use raw int* pointer to safely bypass the ref struct lambda capture rule
    private static unsafe void QuickSortInternal(int* a, int left, int right)
    {
        int length = right - left + 1;

        // 1. ADVANCED SIMD: Intercept chunks of exactly 8 elements
        if (length == 8 && Vector256.IsHardwareAccelerated)
        {
            VectorSort8(a, left);
            return;
        }

        // 2. Micro fallback: Insertion sort
        if (length < InsertMax)
        {
            InsertionSort(a, left, right);
            return;
        }

        int i = left;
        int j = right;
        int part = SelectPivot(a, left, right);

        // Macro splitting loop using direct pointer indexing
        while (i <= j)
        {
            while (a[i] < part)
                i++;
            while (a[j] > part)
                j--;

            if (i <= j)
            {
                int temp = a[i];
                a[i] = a[j];
                a[j] = temp;
                i++;
                j--;
            }
        }

        // 3. PARALLEL ORCHESTRATION:
        // Capturing raw pointers inside lambda expressions is completely valid and bypasses CS9108.
        if (length > ParallelThreshold)
        {
            Parallel.Invoke(
                () =>
                {
                    if (left < j)
                        QuickSortInternal(a, left, j);
                },
                () =>
                {
                    if (i < right)
                        QuickSortInternal(a, i, right);
                }
            );
        }
        else
        {
            if (left < j)
                QuickSortInternal(a, left, j);
            if (i < right)
                QuickSortInternal(a, i, right);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int SelectPivot(int* a, int left, int right)
    {
        int mid = left + ((right - left) >> 1);
        if (a[left] > a[mid])
            Swap(ref a[left], ref a[mid]);
        if (a[left] > a[right])
            Swap(ref a[left], ref a[right]);
        if (a[mid] > a[right])
            Swap(ref a[mid], ref a[right]);
        return a[mid];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Swap(ref int x, ref int y)
    {
        int temp = x;
        x = y;
        y = temp;
    }

    private static unsafe void InsertionSort(int* l, int p, int r)
    {
        for (int j = p + 1; j <= r; j++)
        {
            int key = l[j];
            int i = j - 1;
            while (i - p >= 0 && l[i] > key)
            {
                l[i + 1] = l[i];
                i--;
            }
            l[i + 1] = key;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void VectorSort8(int* a, int index)
    {
        int* start = a + index;

        // Load 8 entries from raw memory directly into the 256-bit vector register
        Vector256<int> v = Vector256.Create(
            start[0],
            start[1],
            start[2],
            start[3],
            start[4],
            start[5],
            start[6],
            start[7]
        );

        // 6-stage hardware sorting network shuffles
        Vector256<int> s1 = Vector256.Shuffle(v, Vector256.Create(1, 0, 3, 2, 5, 4, 7, 6));
        v = Reconstruct(Vector256.Min(v, s1), Vector256.Max(v, s1), 0, 1, 2, 3, 4, 5, 6, 7);

        Vector256<int> s2 = Vector256.Shuffle(v, Vector256.Create(3, 2, 1, 0, 7, 6, 5, 4));
        v = Reconstruct(Vector256.Min(v, s2), Vector256.Max(v, s2), 0, 1, 1, 0, 4, 5, 5, 4);

        Vector256<int> s3 = Vector256.Shuffle(v, Vector256.Create(0, 2, 1, 3, 4, 6, 5, 7));
        v = Reconstruct(Vector256.Min(v, s3), Vector256.Max(v, s3), 0, 2, 1, 3, 4, 6, 5, 7);

        Vector256<int> s4 = Vector256.Shuffle(v, Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0));
        v = Reconstruct(Vector256.Min(v, s4), Vector256.Max(v, s4), 0, 1, 2, 3, 3, 2, 1, 0);

        Vector256<int> s5 = Vector256.Shuffle(v, Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7));
        v = Reconstruct(Vector256.Min(v, s5), Vector256.Max(v, s5), 0, 4, 1, 5, 2, 6, 3, 7);

        Vector256<int> s6 = Vector256.Shuffle(v, Vector256.Create(0, 1, 3, 2, 5, 4, 6, 7));
        v = Reconstruct(Vector256.Min(v, s6), Vector256.Max(v, s6), 0, 1, 3, 2, 5, 4, 6, 7);

        // Write sorted entries straight back to memory
        start[0] = v[0];
        start[1] = v[1];
        start[2] = v[2];
        start[3] = v[3];
        start[4] = v[4];
        start[5] = v[5];
        start[6] = v[6];
        start[7] = v[7];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Reconstruct(
        Vector256<int> low,
        Vector256<int> high,
        int m0,
        int m1,
        int m2,
        int m3,
        int m4,
        int m5,
        int m6,
        int m7
    )
    {
        return Vector256.Create(
            (m0 == 0) ? low[0] : high[0],
            (m1 == 1) ? high[1] : low[1],
            (m2 == 2) ? low[2] : high[2],
            (m3 == 3) ? high[3] : low[3],
            (m4 == 4) ? low[4] : high[4],
            (m5 == 5) ? high[5] : low[5],
            (m6 == 6) ? low[6] : high[6],
            (m7 == 7) ? high[7] : low[7]
        );
    }
}
