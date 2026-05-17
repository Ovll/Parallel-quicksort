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

Console.WriteLine("Starting Hybrid Vectorized QuickSort on 200M items...");
var sw = Stopwatch.StartNew();

OptimizedSort.QuickSort((Span<int>)array);

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

public static class OptimizedSort
{
    private const int InsertMax = 32;

    public static void QuickSort<T>(Span<T> array)
        where T : struct, IComparable<T>
    {
        if (array.Length > 1)
            QuickSortInternal(array, 0, array.Length - 1);
    }

    // Fixed: Added 'struct' constraint so MemoryMarshal.Cast can safely run at compile-time
    private static void QuickSortInternal<T>(Span<T> a, int left, int right)
        where T : struct, IComparable<T>
    {
        int length = right - left + 1;

        // 1. SIMD Interception: If the chunk is exactly 4 ints, sort it via Vector128
        if (length == 4 && typeof(T) == typeof(int))
        {
            VectorizedSortHelper.VectorSort4(MemoryMarshal.Cast<T, int>(a), left);
            return;
        }

        // 2. Standard Fallback: Use Insertion Sort for other micro-partitions under 32 elements
        if (length < InsertMax)
        {
            InsertionSort(a, left, right);
            return;
        }

        int i = left;
        int j = right;
        T part = SelectPivot(a, left, right);

        while (i <= j)
        {
            while (a[i].CompareTo(part) < 0)
                i++;
            while (a[j].CompareTo(part) > 0)
                j--;

            if (i <= j)
            {
                T temp = a[i];
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

    private static T SelectPivot<T>(Span<T> a, int left, int right)
        where T : struct, IComparable<T>
    {
        int mid = left + ((right - left) >> 1);
        if (a[left].CompareTo(a[mid]) > 0)
            Swap(ref a[left], ref a[mid]);
        if (a[left].CompareTo(a[right]) > 0)
            Swap(ref a[left], ref a[right]);
        if (a[mid].CompareTo(a[right]) > 0)
            Swap(ref a[mid], ref a[right]);
        return a[mid];
    }

    private static void Swap<T>(ref T x, ref T y)
        where T : struct
    {
        T temp = x;
        x = y;
        y = temp;
    }

    private static void InsertionSort<T>(Span<T> l, int p, int r)
        where T : struct, IComparable<T>
    {
        for (int j = p + 1; j <= r; j++)
        {
            T key = l[j];
            int i = j - 1;
            while (i - p >= 0 && l[i].CompareTo(key) > 0)
            {
                l[i + 1] = l[i];
                i--;
            }
            l[i + 1] = key;
        }
    }
}

public static class VectorizedSortHelper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorSort4(Span<int> span, int index)
    {
        // Load elements into hardware register
        Vector128<int> v = Vector128.Create(
            span[index],
            span[index + 1],
            span[index + 2],
            span[index + 3]
        );

        // Stage 1: Compare pairs (0,1) and (2,3)
        Vector128<int> shuffled1 = Vector128.Shuffle(v, Vector128.Create(1, 0, 3, 2));
        Vector128<int> low1 = Vector128.Min(v, shuffled1);
        Vector128<int> high1 = Vector128.Max(v, shuffled1);
        v = Vector128.Create(low1[0], high1[1], low1[2], high1[3]);

        // Stage 2: Compare outer/inner pairs (0,2) and (1,3)
        Vector128<int> shuffled2 = Vector128.Shuffle(v, Vector128.Create(2, 3, 0, 1));
        Vector128<int> low2 = Vector128.Min(v, shuffled2);
        Vector128<int> high2 = Vector128.Max(v, shuffled2);
        v = Vector128.Create(low2[0], low2[1], high2[2], high2[3]);

        // Stage 3: Clean up inner pair (1,2)
        Vector128<int> shuffled3 = Vector128.Shuffle(v, Vector128.Create(0, 2, 1, 3));
        Vector128<int> low3 = Vector128.Min(v, shuffled3);
        Vector128<int> high3 = Vector128.Max(v, shuffled3);
        v = Vector128.Create(v[0], low3[1], high3[2], v[3]);

        // Store sorted results back to the span in memory
        span[index] = v[0];
        span[index + 1] = v[1];
        span[index + 2] = v[2];
        span[index + 3] = v[3];
    }
}
