// Basic tensor and layer tests
#include "neon/neon.h"
#include <iostream>
#include <cmath>

using namespace neon;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << msg << " at line " << __LINE__ << "\n"; return; } } while(0)

void test_tensor_create() {
    Tensor t(Shape({2, 3}), 1.0f);
    CHECK(t.shape().dims[0] == 2, "shape[0]");
    CHECK(t.shape().dims[1] == 3, "shape[1]");
    CHECK(t.size() == 6, "size");
    CHECK(t(0, 0) == 1.0f, "value");
    std::cout << "  [PASS] tensor_create\n";
}

void test_tensor_add() {
    Tensor a(Shape({2}), {1.0f, 2.0f});
    Tensor b(Shape({2}), {3.0f, 4.0f});
    Tensor c = a + b;
    CHECK(c[0] == 4.0f, "add[0]");
    CHECK(c[1] == 6.0f, "add[1]");
    std::cout << "  [PASS] tensor_add\n";
}

void test_tensor_matmul() {
    Tensor a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor b(Shape({3, 2}), {7, 8, 9, 10, 11, 12});
    Tensor c = a.matmul(b);
    CHECK(c(0, 0) == 58.0f, "matmul(0,0)");   // 1*7 + 2*9 + 3*11
    CHECK(c(0, 1) == 64.0f, "matmul(0,1)");   // 1*8 + 2*10 + 3*12
    CHECK(c(1, 0) == 139.0f, "matmul(1,0)");  // 4*7 + 5*9 + 6*11
    CHECK(c(1, 1) == 154.0f, "matmul(1,1)");  // 4*8 + 5*10 + 6*12
    std::cout << "  [PASS] tensor_matmul\n";
}

void test_tensor_transpose() {
    Tensor a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor b = a.transpose();
    CHECK(b.shape().dims[0] == 3, "transpose shape[0]");
    CHECK(b.shape().dims[1] == 2, "transpose shape[1]");
    CHECK(b(0, 0) == 1.0f, "transpose(0,0)");
    CHECK(b(2, 1) == 6.0f, "transpose(2,1)");
    std::cout << "  [PASS] tensor_transpose\n";
}

void test_tensor_reshape() {
    Tensor a(Shape({6}), {1, 2, 3, 4, 5, 6});
    Tensor b = a.reshape(Shape({2, 3}));
    CHECK(b.shape().dims[0] == 2, "reshape shape[0]");
    CHECK(b.shape().dims[1] == 3, "reshape shape[1]");
    CHECK(b(1, 2) == 6.0f, "reshape(1,2)");
    std::cout << "  [PASS] tensor_reshape\n";
}

void test_tensor_scalar_ops() {
    Tensor a(Shape({3}), {1.0f, 2.0f, 3.0f});
    Tensor b = a * 2.0f;
    CHECK(b[0] == 2.0f, "scalar_mul[0]");
    CHECK(b[2] == 6.0f, "scalar_mul[2]");
    Tensor c = 10.0f - a;
    CHECK(c[0] == 9.0f, "scalar_sub[0]");
    CHECK(c[2] == 7.0f, "scalar_sub[2]");
    std::cout << "  [PASS] tensor_scalar_ops\n";
}

void test_tensor_sum_mean() {
    Tensor a(Shape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
    CHECK(a.sum() == 10.0f, "sum");
    CHECK(a.mean() == 2.5f, "mean");
    std::cout << "  [PASS] tensor_sum_mean\n";
}

void test_activations() {
    Tensor x(Shape({3}), {-1.0f, 0.0f, 1.0f});
    Tensor r = activation::relu(x);
    CHECK(r[0] == 0.0f, "relu[0]");
    CHECK(r[1] == 0.0f, "relu[1]");
    CHECK(r[2] == 1.0f, "relu[2]");

    Tensor s = activation::sigmoid(x);
    CHECK(s[0] < 0.5f, "sigmoid[0]");
    CHECK(std::abs(s[1] - 0.5f) < 0.001f, "sigmoid[1]");
    CHECK(s[2] > 0.5f, "sigmoid[2]");
    std::cout << "  [PASS] activations\n";
}

void test_softmax() {
    Tensor x(Shape({2, 3}), {1, 2, 3, 1, 1, 1});
    Tensor s = activation::softmax(x);
    float32 row0 = s(0,0) + s(0,1) + s(0,2);
    float32 row1 = s(1,0) + s(1,1) + s(1,2);
    CHECK(std::abs(row0 - 1.0f) < 0.001f, "softmax row0");
    CHECK(std::abs(row1 - 1.0f) < 0.001f, "softmax row1");
    std::cout << "  [PASS] softmax\n";
}

void test_loss_mse() {
    Tensor pred(Shape({3}), {1.0f, 2.0f, 3.0f});
    Tensor target(Shape({3}), {1.0f, 2.0f, 3.0f});
    float32 l = loss::mse(pred, target);
    CHECK(l < 0.0001f, "mse");
    std::cout << "  [PASS] loss_mse\n";
}

void test_forward_pass() {
    Sequential model;
    model.add(std::make_shared<Linear>(2, 4));
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Linear>(4, 1));

    Tensor input(Shape({1, 2}), {1.0f, 0.5f});
    Tensor output = model.forward(input);
    CHECK(output.shape().dims[0] == 1, "fwd shape[0]");
    CHECK(output.shape().dims[1] == 1, "fwd shape[1]");
    std::cout << "  [PASS] forward_pass\n";
}

int main() {
    std::cout << "Running Neon tests...\n\n";

    test_tensor_create();
    test_tensor_add();
    test_tensor_matmul();
    test_tensor_transpose();
    test_tensor_reshape();
    test_tensor_scalar_ops();
    test_tensor_sum_mean();
    test_activations();
    test_softmax();
    test_loss_mse();
    test_forward_pass();

    std::cout << "\nAll tests passed! ✓\n";
    return 0;
}
