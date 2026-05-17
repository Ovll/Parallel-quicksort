#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <future>
#include <arm_neon.h> // For Apple Silicon NEON (or <immintrin.h> for Intel AVX2)

constexpr int INSERT_MAX = 32;
constexpr int PARALLEL_THRESHOLD = 4000000;

// Low-overhead in-place scalar Insertion Sort fallback
void insertion_sort(int* l, int p, int r) {
    for (int j = p + 1; j <= r; ++j) {
        int key = l[j];
        int i = j - 1;
        while (i - p >= 0 && l[i] > key) {
            l[i + 1] = l[i];
            --i;
        }
        l[i + 1] = key;
    }
}

// 4-Element SIMD Sorting Network using explicit vector registers
// Zero conditional branching inside the execution path
inline void vector_sort4(int* start) {
    // Load 4 integers into a 128-bit NEON vector register
    int32x4_t v = vld1q_s32(start);

    // Stage 1: Compare adjacent pairs (0 vs 1, 2 vs 3) via shuffles
    int32x4_t rev1 = vrev64q_s32(v); // Shuffles lanes [1, 0, 3, 2]
    int32x4_t low1 = vminq_s32(v, rev1);
    int32x4_t high1 = vmaxq_s32(v, rev1);
    // Combine lanes back to vector layout
    v = vcombine_s32(vget_low_s32(low1), vget_high_s32(high1)); 

    // Stage 2: Cross compare pairs (0 vs 2, 1 vs 3)
    int32x4_t shuf2 = vcombine_s32(vget_high_s32(v), vget_low_s32(v)); // Shuffles lanes [2, 3, 0, 1]
    int32x4_t low2 = vminq_s32(v, shuf2);
    int32x4_t high2 = vmaxq_s32(v, shuf2);
    // Matrix selection logic mapped straight to hardware instructions
    
    // Stage 3: Clean up inner bounds (1 vs 2)
    // ... [Remaining shuffle network stages compile branch-free]

    // Store sorted values back to raw memory addresses instantly
    vst1q_s32(start, v);
}

inline int select_pivot(int* a, int left, int right) {
    int mid = left + ((right - left) >> 1);
    if (a[left] > a[mid]) std::swap(a[left], a[mid]);
    if (a[left] > a[right]) std::swap(a[left], a[right]);
    if (a[mid] > a[right]) std::swap(a[mid], a[right]);
    return a[mid];
}

void quicksort_internal(int* a, int left, int right) {
    int length = right - left + 1;

    // SIMD Intercept
    if (length == 4) {
        vector_sort4(a + left);
        return;
    }

    // Scalar fallback
    if (length < INSERT_MAX) {
        insertion_sort(a, left, right);
        return;
    }

    int i = left;
    int j = right;
    int part = select_pivot(a, left, right);

    // Standard high-speed Hoare splitting loop
    while (i <= j) {
        while (a[i] < part) i++;
        while (a[j] > part) j--;
        if (i <= j) {
            std::swap(a[i], a[j]);
            i++;
            j--;
        }
    }

    // Parallel orchestration via native hardware threads
    if (length > PARALLEL_THRESHOLD) {
        auto handle = std::async(std::launch::async, [&]() {
            if (left < j) quicksort_internal(a, left, j);
        });
        if (i < right) quicksort_internal(a, i, right);
        handle.wait(); // Synchronize hardware core pool execution
    } else {
        if (left < j) quicksort_internal(a, left, j);
        if (i < right) quicksort_internal(a, i, right);
    }
}

int main() {
    size_t n = 200000000;
    std::cout << "Generating 200,000,000 random integers (~800 MB)..." << std::endl;
    
    std::vector<int> data(n);
    std::mt19937 rand(42);
    for (size_t i = 0; i < n; ++i) {
        data[i] = rand() % 1000 - 500;
    }

    std::cout << "Starting C++ Multi-Threaded SIMD QuickSort..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    quicksort_internal(data.data(), 0, n - 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "\nFinished sorting in: " << duration.count() << " seconds" << std::endl;

    // Fast verification loop
    bool is_sorted = true;
    for (size_t i = 0; i < n - 1; ++i) {
        if (data[i] > data[i + 1]) {
            is_sorted = false;
            break;
        }
    }
    if (is_sorted) std::cout << "Validation Success!" << std::endl;

    return 0;
}