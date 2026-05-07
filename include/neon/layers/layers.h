#pragma once

#include "neon/core/tensor.h"
#include "neon/core/activations.h"
#include <memory>
#include <vector>

namespace neon {

// ─── Layer Base ─────────────────────────────────────────────────
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; }
    virtual std::vector<Tensor*> gradients() { return {}; }
    virtual void train() { training_ = true; }
    virtual void eval() { training_ = false; }
    virtual std::string name() const = 0;

protected:
    bool training_ = true;
    Tensor input_cache_;
};

using LayerPtr = std::shared_ptr<Layer>;

// ─── Linear (Fully Connected) ──────────────────────────────────
class Linear : public Layer {
public:
    Linear(size_t in_features, size_t out_features, unsigned seed = 42)
        : in_features_(in_features)
        , out_features_(out_features)
    {
        weights_ = Tensor::xavier_uniform(Shape({in_features, out_features}), in_features, out_features, seed);
        bias_ = Tensor::zeros(Shape({out_features}));
    }

    Tensor forward(const Tensor& input) override {
        input_cache_ = input;
        Tensor out = input.matmul(weights_);
        // Broadcast bias over batch dimension
        for (size_t i = 0; i < out.shape().dims[0]; ++i)
            for (size_t j = 0; j < out_features_; ++j)
                out(i, j) += bias_[j];
        return out;
    }

    Tensor backward(const Tensor& grad_output) override {
        // grad_weights = input^T @ grad_output  (in x batch) @ (batch x out) = (in x out)
        Tensor input_t = input_cache_.transpose();
        grad_weights_ = input_t.matmul(grad_output);

        // grad_bias = sum over batch dimension
        size_t batch = grad_output.shape().dims[0];
        grad_bias_ = Tensor(Shape({out_features_}));
        for (size_t j = 0; j < out_features_; ++j) {
            float32 s = 0;
            for (size_t i = 0; i < batch; ++i)
                s += grad_output(i, j);
            grad_bias_[j] = s;
        }

        // grad_input = grad_output @ weights^T  (batch x out) @ (out x in) = (batch x in)
        return grad_output.matmul(weights_.transpose());
    }

    std::vector<Tensor*> parameters() override {
        return {&weights_, &bias_};
    }

    std::vector<Tensor*> gradients() override {
        return {&grad_weights_, &grad_bias_};
    }

    std::string name() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + std::to_string(out_features_) + ")";
    }

private:
    size_t in_features_, out_features_;
    Tensor weights_, bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
};

// ─── ReLU ───────────────────────────────────────────────────────
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = input;
        return activation::relu(input);
    }

    Tensor backward(const Tensor& grad_output) override {
        return grad_output * activation::relu_grad(input_cache_);
    }

    std::string name() const override { return "ReLU"; }
};

// ─── Sigmoid ───────────────────────────────────────────────────
class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = activation::sigmoid(input);
        return input_cache_;
    }

    Tensor backward(const Tensor& grad_output) override {
        // input_cache_ = sigmoid(x), so sigmoid_grad = s * (1 - s)
        Tensor grad(input_cache_.shape());
        for (size_t i = 0; i < input_cache_.size(); ++i) {
            float32 s = input_cache_[i];
            grad[i] = s * (1.0f - s);
        }
        return grad_output * grad;
    }

    std::string name() const override { return "Sigmoid"; }
};

// ─── Tanh ──────────────────────────────────────────────────────
class Tanh : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = activation::tanh_act(input);
        return input_cache_;
    }

    Tensor backward(const Tensor& grad_output) override {
        // input_cache_ = tanh(x), so tanh_grad = 1 - t^2
        Tensor grad(input_cache_.shape());
        for (size_t i = 0; i < input_cache_.size(); ++i) {
            float32 t = input_cache_[i];
            grad[i] = 1.0f - t * t;
        }
        return grad_output * grad;
    }

    std::string name() const override { return "Tanh"; }
};

// ─── Softmax ───────────────────────────────────────────────────
class Softmax : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        input_cache_ = activation::softmax(input);
        return input_cache_;
    }

    Tensor backward(const Tensor& grad_output) override {
        return grad_output;
    }

    std::string name() const override { return "Softmax"; }
};

// ─── Dropout ───────────────────────────────────────────────────
class Dropout : public Layer {
public:
    explicit Dropout(float32 p = 0.5f) : p_(p) {}

    Tensor forward(const Tensor& input) override {
        if (!training_) return input;
        mask_ = Tensor::randu(input.shape());
        for (size_t i = 0; i < mask_.size(); ++i)
            mask_[i] = (mask_[i] > p_) ? (1.0f / (1.0f - p_)) : 0.0f;
        return input * mask_;
    }

    Tensor backward(const Tensor& grad_output) override {
        return grad_output * mask_;
    }

    std::string name() const override {
        return "Dropout(p=" + std::to_string(p_) + ")";
    }

private:
    float32 p_;
    Tensor mask_;
};

// ─── Sequential ────────────────────────────────────────────────
class Sequential : public Layer {
public:
    Sequential() = default;
    explicit Sequential(std::vector<LayerPtr> layers) : layers_(std::move(layers)) {}

    void add(LayerPtr layer) { layers_.push_back(std::move(layer)); }

    Tensor forward(const Tensor& input) override {
        Tensor x = input;
        for (auto& layer : layers_) {
            x = layer->forward(x);
        }
        return x;
    }

    Tensor backward(const Tensor& grad_output) override {
        Tensor grad = grad_output;
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            grad = layers_[i]->backward(grad);
        }
        return grad;
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> params;
        for (auto& layer : layers_) {
            auto lp = layer->parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }

    std::vector<Tensor*> gradients() override {
        std::vector<Tensor*> grads;
        for (auto& layer : layers_) {
            auto lg = layer->gradients();
            grads.insert(grads.end(), lg.begin(), lg.end());
        }
        return grads;
    }

    void train() override {
        Layer::train();
        for (auto& l : layers_) l->train();
    }

    void eval() override {
        Layer::eval();
        for (auto& l : layers_) l->eval();
    }

    std::string name() const override {
        std::string s = "Sequential[\n";
        for (auto& l : layers_) s += "  " + l->name() + "\n";
        s += "]";
        return s;
    }

    const std::vector<LayerPtr>& layers() const { return layers_; }

private:
    std::vector<LayerPtr> layers_;
};

} // namespace neon
