#pragma once

#include "neon/core/tensor.h"
#include "neon/layers/layers.h"
#include <vector>
#include <cmath>

namespace neon {

// ─── Optimizer Base ─────────────────────────────────────────────
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual std::string name() const = 0;

protected:
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
};

// ─── SGD ────────────────────────────────────────────────────────
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, std::vector<Tensor*> grads,
        float32 lr = 0.01f, float32 momentum = 0.0f, float32 weight_decay = 0.0f)
        : lr_(lr)
        , momentum_(momentum)
        , weight_decay_(weight_decay)
        , params_(std::move(params))
        , grads_(std::move(grads))
    {
        // Initialize velocity buffers
        for (auto* p : params_) {
            velocity_.emplace_back(p->shape(), 0.0f);
        }
    }

    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            Tensor* p = params_[i];
            Tensor* g = grads_[i];

            // Weight decay (L2 regularization)
            if (weight_decay_ > 0) {
                for (size_t j = 0; j < g->size(); ++j)
                    (*g)[j] += weight_decay_ * (*p)[j];
            }

            // Momentum update
            if (momentum_ > 0) {
                velocity_[i] = velocity_[i] * momentum_ - (*g) * lr_;
                *p += velocity_[i];
            } else {
                *p -= (*g) * lr_;
            }
        }
    }

    void zero_grad() override {
        for (auto* g : grads_) g->zero_();
        for (auto& v : velocity_) v.zero_();
    }

    std::string name() const override {
        return "SGD(lr=" + std::to_string(lr_) +
               ", momentum=" + std::to_string(momentum_) + ")";
    }

private:
    float32 lr_, momentum_, weight_decay_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    std::vector<Tensor> velocity_;
};

// ─── Adam ───────────────────────────────────────────────────────
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, std::vector<Tensor*> grads,
         float32 lr = 0.001f, float32 beta1 = 0.9f, float32 beta2 = 0.999f,
         float32 eps = 1e-8f, float32 weight_decay = 0.0f)
        : lr_(lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , eps_(eps)
        , weight_decay_(weight_decay)
        , t_(0)
        , params_(std::move(params))
        , grads_(std::move(grads))
    {
        for (auto* p : params_) {
            m_.emplace_back(p->shape(), 0.0f);  // first moment
            v_.emplace_back(p->shape(), 0.0f);  // second moment
        }
    }

    void step() override {
        ++t_;
        float32 bc1 = 1.0f - std::pow(beta1_, t_);
        float32 bc2 = 1.0f - std::pow(beta2_, t_);

        for (size_t i = 0; i < params_.size(); ++i) {
            Tensor* p = params_[i];
            Tensor* g = grads_[i];

            // Weight decay
            Tensor grad = weight_decay_ > 0 ? (*g) + (*p) * weight_decay_ : *g;

            // Update moments
            m_[i] = m_[i] * beta1_ + grad * (1.0f - beta1_);
            v_[i] = v_[i] * beta2_ + (grad * grad) * (1.0f - beta2_);

            // Bias-corrected estimates
            Tensor m_hat = m_[i] * (1.0f / bc1);
            Tensor v_hat = v_[i] * (1.0f / bc2);

            // Update parameters
            for (size_t j = 0; j < p->size(); ++j) {
                (*p)[j] -= lr_ * m_hat[j] / (std::sqrt(v_hat[j]) + eps_);
            }
        }
    }

    void zero_grad() override {
        for (auto* g : grads_) g->zero_();
    }

    std::string name() const override {
        return "Adam(lr=" + std::to_string(lr_) + ")";
    }

private:
    float32 lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    std::vector<Tensor> m_, v_;
};

// ─── Helper: create optimizer from a Sequential model ───────────
inline std::unique_ptr<Adam> make_adam(Sequential& model, float32 lr = 0.001f) {
    return std::make_unique<Adam>(model.parameters(), model.gradients(), lr);
}

inline std::unique_ptr<SGD> make_sgd(Sequential& model, float32 lr = 0.01f, float32 momentum = 0.9f) {
    return std::make_unique<SGD>(model.parameters(), model.gradients(), lr, momentum);
}

} // namespace neon
