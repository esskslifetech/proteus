#include <proteus/kernels/simd_stats.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace proteus::kernels {

// The implementations are in the header as templates
// This file exists for explicit instantiation if needed

// Explicit instantiations for common types
template double mean_simd<float>(const float*, std::size_t);
template double mean_simd<double>(const double*, std::size_t);

template double variance_simd<float>(const float*, std::size_t);
template double variance_simd<double>(const double*, std::size_t);

template double stddev_simd<float>(const float*, std::size_t);
template double stddev_simd<double>(const double*, std::size_t);

template float quantile_simd<float>(float*, std::size_t, double);
template double quantile_simd<double>(double*, std::size_t, double);

template double correlation_simd<float>(const float*, const float*, std::size_t);
template double correlation_simd<double>(const double*, const double*, std::size_t);

template std::vector<std::size_t> histogram_simd<float>(const float*, std::size_t, float, float, std::size_t);
template std::vector<std::size_t> histogram_simd<double>(const double*, std::size_t, double, double, std::size_t);

template void gemv_simd<float>(const float*, const float*, float*, std::size_t, std::size_t);
template void gemv_simd<double>(const double*, const double*, double*, std::size_t, std::size_t);

template void gemm_simd<float>(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
template void gemm_simd<double>(const double*, const double*, double*, std::size_t, std::size_t, std::size_t);

template double euclidean_distance_simd<float>(const float*, const float*, std::size_t);
template double euclidean_distance_simd<double>(const double*, const double*, std::size_t);

template double manhattan_distance_simd<float>(const float*, const float*, std::size_t);
template double manhattan_distance_simd<double>(const double*, const double*, std::size_t);

template double cosine_similarity_simd<float>(const float*, const float*, std::size_t);
template double cosine_similarity_simd<double>(const double*, const double*, std::size_t);

} // namespace proteus::kernels