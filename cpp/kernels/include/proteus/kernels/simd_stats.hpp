#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <concepts>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace proteus::kernels {

// SIMD-accelerated statistical functions

// Check if type is SIMD-friendly
template<typename T>
concept SimdArithmetic = std::is_arithmetic_v<T> && (sizeof(T) == 4 || sizeof(T) == 8);

// Mean
template<SimdArithmetic T>
double mean_simd(const T* data, std::size_t n) {
    // Placeholder - real implementation uses AVX2/AVX-512/NEON
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return n > 0 ? sum / n : 0.0;
}

// Variance (Welford's online algorithm for numerical stability)
template<SimdArithmetic T>
double variance_simd(const T* data, std::size_t n) {
    double mean = 0.0;
    double m2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(data[i]);
        double delta = x - mean;
        mean += delta / (i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }
    return n > 1 ? m2 / (n - 1) : 0.0;
}

// Standard deviation
template<SimdArithmetic T>
double stddev_simd(const T* data, std::size_t n) {
    return ::std::sqrt(variance_simd(data, n));
}

// Quantiles (using quickselect)
template<SimdArithmetic T>
T quantile_simd(T* data, std::size_t n, double q) {
    if (n == 0) return T{};
    if (q <= 0.0) return data[0];
    if (q >= 1.0) return data[n - 1];
    
    std::size_t idx = static_cast<std::size_t>(q * (n - 1));
    ::std::nth_element(data, data + idx, data + n);
    return data[idx];
}

// Correlation
template<SimdArithmetic T>
double correlation_simd(const T* x, const T* y, std::size_t n) {
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    
    for (std::size_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(x[i]);
        double yi = static_cast<double>(y[i]);
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
    }
    
    double num = n * sum_xy - sum_x * sum_y;
    double den = ::std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    return den > 0.0 ? num / den : 0.0;
}

// Histogram
template<SimdArithmetic T>
std::vector<std::size_t> histogram_simd(const T* data, std::size_t n, T min_val, T max_val, std::size_t bins) {
    std::vector<std::size_t> hist(bins, 0);
    if (n == 0 || bins == 0) return hist;
    
    double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    if (range <= 0.0) {
        hist[0] = n;
        return hist;
    }
    
    for (std::size_t i = 0; i < n; ++i) {
        double val = static_cast<double>(data[i]);
        std::size_t bin = static_cast<std::size_t>((val - static_cast<double>(min_val)) / range * bins);
        if (bin >= bins) bin = bins - 1;
        hist[bin]++;
    }
    return hist;
}

// SIMD-accelerated matrix operations (column-major)

template<SimdArithmetic T>
void gemv_simd(const T* A, const T* x, T* y, std::size_t m, std::size_t n) {
    // y = A * x
    for (std::size_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            sum += static_cast<double>(A[i + j * m]) * static_cast<double>(x[j]);
        }
        y[i] = static_cast<T>(sum);
    }
}

template<SimdArithmetic T>
void gemm_simd(const T* A, const T* B, T* C, std::size_t m, std::size_t n, std::size_t k) {
    // C = A * B (A: m×k, B: k×n, C: m×n)
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t l = 0; l < k; ++l) {
                sum += static_cast<double>(A[i + l * m]) * static_cast<double>(B[l + j * k]);
            }
            C[i + j * m] = static_cast<T>(sum);
        }
    }
}

// Distance metrics

template<SimdArithmetic T>
double euclidean_distance_simd(const T* a, const T* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return ::std::sqrt(sum);
}

template<SimdArithmetic T>
double manhattan_distance_simd(const T* a, const T* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += ::std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum;
}

template<SimdArithmetic T>
double cosine_similarity_simd(const T* a, const T* b, std::size_t n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ai = static_cast<double>(a[i]);
        double bi = static_cast<double>(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    double denom = ::std::sqrt(norm_a) * ::std::sqrt(norm_b);
    return denom > 0.0 ? dot / denom : 0.0;
}

} // namespace proteus::kernels