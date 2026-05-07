#pragma once

#include "neon/core/tensor.h"

namespace neon {

// ─── Activation Functions ───────────────────────────────────────
namespace activation {

    inline Tensor relu(const Tensor& x) {
        return x.apply([](float32 v) { return v > 0 ? v : 0; });
    }

    inline Tensor relu_grad(const Tensor& x) {
        return x.apply([](float32 v) { return v > 0 ? 1.0f : 0.0f; });
    }

    inline Tensor sigmoid(const Tensor& x) {
        return x.apply([](float32 v) { return 1.0f / (1.0f + std::exp(-v)); });
    }

    inline Tensor sigmoid_grad(const Tensor& x) {
        return x.apply([](float32 v) {
            float32 s = 1.0f / (1.0f + std::exp(-v));
            return s * (1.0f - s);
        });
    }

    inline Tensor tanh_act(const Tensor& x) {
        return x.apply([](float32 v) { return std::tanh(v); });
    }

    inline Tensor tanh_grad(const Tensor& x) {
        return x.apply([](float32 v) {
            float32 t = std::tanh(v);
            return 1.0f - t * t;
        });
    }

    inline Tensor softmax(const Tensor& x) {
        NEON_ASSERT(x.shape().ndim() == 2, "softmax requires 2D tensor");
        size_t rows = x.shape().dims[0], cols = x.shape().dims[1];
        Tensor result(x.shape());
        for (size_t i = 0; i < rows; ++i) {
            float32 max_val = x[i * cols];
            for (size_t j = 1; j < cols; ++j)
                max_val = std::max(max_val, x[i * cols + j]);
            float32 sum = 0;
            for (size_t j = 0; j < cols; ++j) {
                result[i * cols + j] = std::exp(x[i * cols + j] - max_val);
                sum += result[i * cols + j];
            }
            for (size_t j = 0; j < cols; ++j)
                result[i * cols + j] /= sum;
        }
        return result;
    }

    inline Tensor leaky_relu(const Tensor& x, float32 alpha = 0.01f) {
        return x.apply([alpha](float32 v) { return v > 0 ? v : alpha * v; });
    }

    inline Tensor leaky_relu_grad(const Tensor& x, float32 alpha = 0.01f) {
        return x.apply([alpha](float32 v) { return v > 0 ? 1.0f : alpha; });
    }

} // namespace activation

} // namespace neon
