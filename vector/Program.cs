using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

Console.WriteLine("Generating 200,000,000 random integers (~800 MB)...");
var rand = new Random(42);
int[] array = new int[200_000_000];
for (int i = 0; i < array.Length; i++)
{
    array[i] = rand.Next(-500, 500);
}

Console.WriteLine("Starting Ultra-Optimized Vector256 QuickSort on 200M items... ");
var sw = Stopwatch.StartNew();

UltraSort.QuickSort(array);

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
    Console.WriteLine("Validation Success! 200M items are perfectly sorted.");
}

public static class UltraSort
{
    private const int InsertMax = 32;

    public static void QuickSort(Span<int> array)
    {
        if (array.Length > 1)
            QuickSortInternal(array, 0, array.Length - 1);
    }

    private static void QuickSortInternal(Span<int> a, int left, int right)
    {
        int length = right - left + 1;

        // 1. ADVANCED SIMD: Intercept chunks of exactly 8 elements
        if (length == 8 && Vector256.IsHardwareAccelerated)
        {
            VectorSort8(a, left);
            return;
        }

        // 2. Fallback to scalar insertion sort for other micro-partitions
        if (length < InsertMax)
        {
            InsertionSort(a, left, right);
            return;
        }

        int i = left;
        int j = right;
        int part = SelectPivot(a, left, right);

        // Primitive integer comparisons compile down directly to raw native assembly commands
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

        if (left < j)
            QuickSortInternal(a, left, j);
        if (i < right)
            QuickSortInternal(a, i, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SelectPivot(Span<int> a, int left, int right)
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

    private static void InsertionSort(Span<int> l, int p, int r)
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

    /// <summary>
    /// Vectorized 8-element sorting network using 256-bit registers (Vector256).
    /// Zero conditional branches.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void VectorSort8(Span<int> span, int index)
    {
        // Grab a direct reference to the memory block origin to perform zero-overhead indexing
        ref int src = ref MemoryMarshal.GetReference(span);
        ref int start = ref Unsafe.Add(ref src, index);

        // Load 8 elements directly into the 256-bit vector register using native pointer scaling
        Vector256<int> v = Vector256.Create(
            start,
            Unsafe.Add(ref start, 1),
            Unsafe.Add(ref start, 2),
            Unsafe.Add(ref start, 3),
            Unsafe.Add(ref start, 4),
            Unsafe.Add(ref start, 5),
            Unsafe.Add(ref start, 6),
            Unsafe.Add(ref start, 7)
        );

        // Stage 1
        Vector256<int> s1 = Vector256.Shuffle(v, Vector256.Create(1, 0, 3, 2, 5, 4, 7, 6));
        v = Reconstruct(Vector256.Min(v, s1), Vector256.Max(v, s1), 0, 1, 2, 3, 4, 5, 6, 7);

        // Stage 2
        Vector256<int> s2 = Vector256.Shuffle(v, Vector256.Create(3, 2, 1, 0, 7, 6, 5, 4));
        v = Reconstruct(Vector256.Min(v, s2), Vector256.Max(v, s2), 0, 1, 1, 0, 4, 5, 5, 4);

        // Stage 3
        Vector256<int> s3 = Vector256.Shuffle(v, Vector256.Create(0, 2, 1, 3, 4, 6, 5, 7));
        v = Reconstruct(Vector256.Min(v, s3), Vector256.Max(v, s3), 0, 2, 1, 3, 4, 6, 5, 7);

        // Stage 4
        Vector256<int> s4 = Vector256.Shuffle(v, Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0));
        v = Reconstruct(Vector256.Min(v, s4), Vector256.Max(v, s4), 0, 1, 2, 3, 3, 2, 1, 0);

        // Stage 5
        Vector256<int> s5 = Vector256.Shuffle(v, Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7));
        v = Reconstruct(Vector256.Min(v, s5), Vector256.Max(v, s5), 0, 4, 1, 5, 2, 6, 3, 7);

        // Stage 6
        Vector256<int> s6 = Vector256.Shuffle(v, Vector256.Create(0, 1, 3, 2, 5, 4, 6, 7));
        v = Reconstruct(Vector256.Min(v, s6), Vector256.Max(v, s6), 0, 1, 3, 2, 5, 4, 6, 7);

        // Store sorted results back to memory using direct safe pointer offsets
        start = v[0];
        Unsafe.Add(ref start, 1) = v[1];
        Unsafe.Add(ref start, 2) = v[2];
        Unsafe.Add(ref start, 3) = v[3];
        Unsafe.Add(ref start, 4) = v[4];
        Unsafe.Add(ref start, 5) = v[5];
        Unsafe.Add(ref start, 6) = v[6];
        Unsafe.Add(ref start, 7) = v[7];
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
