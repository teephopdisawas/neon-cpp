#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cmath>

// Use NDEBUG to disable asserts in release
#ifndef NDEBUG
  #define NEON_ASSERT(cond, msg) \
    do { if (!(cond)) { throw std::runtime_error("Neon assert: " msg); } } while(0)
#else
  #define NEON_ASSERT(cond, msg) ((void)0)
#endif

namespace neon {

// ─── Type aliases ───────────────────────────────────────────────
using float32 = float;
using float64 = double;

// ─── Shape helper ──────────────────────────────────────────────
struct Shape {
    std::vector<size_t> dims;

    Shape() = default;
    Shape(std::initializer_list<size_t> d) : dims(d) {}
    Shape(std::vector<size_t> d) : dims(std::move(d)) {}

    size_t ndim() const { return dims.size(); }
    size_t total() const {
        if (dims.empty()) return 0;
        return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
    }

    bool operator==(const Shape& o) const { return dims == o.dims; }
    bool operator!=(const Shape& o) const { return dims != o.dims; }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << dims[i];
        }
        ss << ")";
        return ss.str();
    }
};

// ─── Tensor ─────────────────────────────────────────────────────
class Tensor {
public:
    // Default: empty scalar-like
    Tensor() = default;

    // Shape only — zero-filled
    explicit Tensor(const Shape& shape)
        : shape_(shape)
        , data_(shape_.total(), 0.0f)
    {}

    // Shape + fill value
    Tensor(const Shape& shape, float32 fill)
        : shape_(shape)
        , data_(shape_.total(), fill)
    {}

    // Shape + data
    Tensor(const Shape& shape, const std::vector<float32>& data)
        : shape_(shape)
        , data_(data)
    {
        if (data_.size() != shape_.total()) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
    }

    // Shape + data (move)
    Tensor(const Shape& shape, std::vector<float32>&& data)
        : shape_(shape)
        , data_(std::move(data))
    {
        if (data_.size() != shape_.total()) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
    }

    // Static factory methods
    static Tensor zeros(const Shape& shape) { return Tensor(shape, 0.0f); }
    static Tensor ones(const Shape& shape) { return Tensor(shape, 1.0f); }

    static Tensor randn(const Shape& shape, float32 mean = 0.0f, float32 stddev = 1.0f, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<float32> dist(mean, stddev);
        std::vector<float32> data(shape.total());
        std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
        return Tensor(shape, std::move(data));
    }

    static Tensor randu(const Shape& shape, float32 lo = 0.0f, float32 hi = 1.0f, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float32> dist(lo, hi);
        std::vector<float32> data(shape.total());
        std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
        return Tensor(shape, std::move(data));
    }

    static Tensor xavier_uniform(const Shape& shape, size_t fan_in, size_t fan_out, unsigned seed = 42) {
        float32 limit = std::sqrt(6.0f / (fan_in + fan_out));
        return randu(shape, -limit, limit, seed);
    }

    // Accessors
    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t size() const { return data_.size(); }
    const std::vector<float32>& data() const { return data_; }
    std::vector<float32>& data() { return data_; }

    // Flat index access
    float32& operator[](size_t i) { return data_[i]; }
    float32 operator[](size_t i) const { return data_[i]; }

    float32& at(size_t i) { return data_.at(i); }
    float32 at(size_t i) const { return data_.at(i); }

    // 2D convenience access: (row, col)
    float32& operator()(size_t row, size_t col) {
        return data_[row * shape_.dims[1] + col];
    }
    float32 operator()(size_t row, size_t col) const {
        return data_[row * shape_.dims[1] + col];
    }

    // Raw pointer access (for BLAS interop)
    float32* ptr() { return data_.data(); }
    const float32* ptr() const { return data_.data(); }

    // Reshape
    Tensor reshape(const Shape& new_shape) const {
        if (new_shape.total() != shape_.total()) {
            throw std::invalid_argument("Reshape: total size mismatch");
        }
        return Tensor(new_shape, data_);
    }

    // Element-wise operations
    Tensor operator+(const Tensor& o) const { return binary_op(o, std::plus<float32>()); }
    Tensor operator-(const Tensor& o) const { return binary_op(o, std::minus<float32>()); }
    Tensor operator*(const Tensor& o) const { return binary_op(o, std::multiplies<float32>()); }
    Tensor operator/(const Tensor& o) const { return binary_op(o, std::divides<float32>()); }

    Tensor operator+(float32 s) const { return unary_op([s](float32 x) { return x + s; }); }
    Tensor operator-(float32 s) const { return unary_op([s](float32 x) { return x - s; }); }
    Tensor operator*(float32 s) const { return unary_op([s](float32 x) { return x * s; }); }
    Tensor operator/(float32 s) const { return unary_op([s](float32 x) { return x / s; }); }

    // Matrix multiplication (2D only)
    Tensor matmul(const Tensor& o) const {
        NEON_ASSERT(shape_.ndim() == 2 && o.shape_.ndim() == 2, "matmul requires 2D tensors");
        NEON_ASSERT(shape_.dims[1] == o.shape_.dims[0], "matmul inner dims mismatch");
        size_t m = shape_.dims[0], k = shape_.dims[1], n = o.shape_.dims[1];
        Tensor result(Shape({m, n}));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < k; ++p)
                    result[i * n + j] += data_[i * k + p] * o.data_[p * n + j];
        return result;
    }

    // Transpose (2D)
    Tensor transpose() const {
        NEON_ASSERT(shape_.ndim() == 2, "transpose requires 2D tensor");
        size_t m = shape_.dims[0], n = shape_.dims[1];
        Tensor result(Shape({n, m}));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                result[j * m + i] = data_[i * n + j];
        return result;
    }

    // Reduce operations
    float32 sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0.0f);
    }

    float32 mean() const {
        return sum() / static_cast<float32>(data_.size());
    }

    // Apply functions
    Tensor apply(std::function<float32(float32)> fn) const {
        return unary_op(std::move(fn));
    }

    // In-place operations
    Tensor& operator+=(const Tensor& o) {
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
        return *this;
    }
    Tensor& operator-=(const Tensor& o) {
        for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
        return *this;
    }
    Tensor& operator*=(float32 s) {
        for (auto& v : data_) v *= s;
        return *this;
    }
    Tensor& operator/=(float32 s) {
        for (auto& v : data_) v /= s;
        return *this;
    }

    // Fill
    Tensor& fill_(float32 v) { std::fill(data_.begin(), data_.end(), v); return *this; }
    Tensor& zero_() { return fill_(0.0f); }

    // Clone
    Tensor clone() const { return Tensor(shape_, data_); }

    // Print
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor" << t.shape_.to_string() << " ";
        os << "[";
        for (size_t i = 0; i < std::min(t.data_.size(), size_t(20)); ++i) {
            if (i > 0) os << ", ";
            os << t.data_[i];
        }
        if (t.data_.size() > 20) os << ", ...";
        os << "]";
        return os;
    }

private:
    Shape shape_;
    std::vector<float32> data_;

    Tensor binary_op(const Tensor& o, std::function<float32(float32, float32)> op) const {
        if (shape_.total() == 1) {
            Tensor result(o.shape_);
            for (size_t i = 0; i < o.data_.size(); ++i)
                result.data_[i] = op(data_[0], o.data_[i]);
            return result;
        }
        if (o.shape_.total() == 1) {
            Tensor result(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
                result.data_[i] = op(data_[i], o.data_[0]);
            return result;
        }
        if (shape_ != o.shape_) {
            throw std::invalid_argument("Shape mismatch: " + shape_.to_string() + " vs " + o.shape_.to_string());
        }
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = op(data_[i], o.data_[i]);
        return result;
    }

    Tensor unary_op(std::function<float32(float32)> op) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = op(data_[i]);
        return result;
    }
};

// Scalar on left
inline Tensor operator+(float32 s, const Tensor& t) { return t + s; }
inline Tensor operator*(float32 s, const Tensor& t) { return t * s; }
inline Tensor operator-(float32 s, const Tensor& t) { return Tensor(t.shape(), s) - t; }

} // namespace neon
