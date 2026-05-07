#pragma once

#include "neon/core/tensor.h"
#include "neon/core/activations.h"
#include <cmath>

namespace neon {

// ─── Loss Functions ─────────────────────────────────────────────
namespace loss {

    // Mean Squared Error
    inline float32 mse(const Tensor& pred, const Tensor& target) {
        Tensor diff = pred - target;
        Tensor sq = diff * diff;
        return sq.mean();
    }

    inline Tensor mse_grad(const Tensor& pred, const Tensor& target) {
        float32 n = static_cast<float32>(pred.size());
        return (pred - target) * (2.0f / n);
    }

    // Binary Cross-Entropy
    inline float32 binary_cross_entropy(const Tensor& pred, const Tensor& target) {
        float32 eps = 1e-7f;
        float32 loss = 0;
        for (size_t i = 0; i < pred.size(); ++i) {
            float32 p = std::clamp(pred[i], eps, 1.0f - eps);
            loss += -(target[i] * std::log(p) + (1.0f - target[i]) * std::log(1.0f - p));
        }
        return loss / static_cast<float32>(pred.size());
    }

    inline Tensor binary_cross_entropy_grad(const Tensor& pred, const Tensor& target) {
        float32 eps = 1e-7f;
        Tensor grad(pred.shape());
        float32 n = static_cast<float32>(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            float32 p = std::clamp(pred[i], eps, 1.0f - eps);
            grad[i] = -(target[i] / p - (1.0f - target[i]) / (1.0f - p)) / n;
        }
        return grad;
    }

    // Categorical Cross-Entropy (expects softmax probabilities)
    inline float32 cross_entropy(const Tensor& pred, const Tensor& target) {
        float32 eps = 1e-7f;
        float32 loss = 0;
        for (size_t i = 0; i < pred.size(); ++i) {
            float32 p = std::clamp(pred[i], eps, 1.0f);
            loss += -target[i] * std::log(p);
        }
        return loss / static_cast<float32>(pred.shape().dims[0]);
    }

    inline Tensor cross_entropy_grad(const Tensor& pred, const Tensor& target) {
        float32 n = static_cast<float32>(pred.shape().dims[0]);
        return (pred - target) / n;
    }

} // namespace loss

} // namespace neon
