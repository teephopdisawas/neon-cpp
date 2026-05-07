# 🔶 Neon AI Framework

A lightweight, header-only C++ neural network framework with a web-based GUI and Docker support.

Neon provides tensor operations, layers, loss functions, and optimizers — all in modern C++17 with zero external dependencies at compile time. Train neural networks from your browser.

## Features

- **Header-only library** — just `#include "neon/neon.h"`
- **Web GUI** — configure, train, and visualise models from your browser
- **REST API** — JSON-based API for programmatic access
- **Docker support** — runs in a container with `docker compose up`
- **Built-in datasets** — XOR and Circle classification out of the box
- **Layers** — Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, Sequential
- **Losses** — MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers** — SGD (with momentum), Adam

## Quick Start

### Option 1: Docker (easiest)

```bash
docker compose up --build
```

Open http://localhost:8080 in your browser. Done.

### Option 2: Build from source

```bash
git clone https://github.com/teephopdisawas/neon-cpp.git
cd neon-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run the web server
./neon_server 8080
```

Open http://localhost:8080 in your browser.

### Option 3: Run the XOR example directly

```bash
cd build
./xor_example
```

## Project Structure

```
neon/
├── CMakeLists.txt              # Build system
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # One-command deployment
├── include/neon/
│   ├── neon.h                  # Single-include header
│   ├── core/
│   │   ├── tensor.h            # N-dim tensor with ops
│   │   └── activations.h       # ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
│   ├── layers/
│   │   └── layers.h            # Linear, activations, Dropout, Sequential
│   ├── loss/
│   │   └── loss.h              # MSE, Binary CE, Categorical CE
│   ├── optimizers/
│   │   └── optimizers.h        # SGD, Adam
│   └── server.h                # REST API server
├── src/
│   └── server_main.cpp         # Server entry point
├── examples/
│   └── xor_example.cpp         # XOR neural net demo
├── tests/
│   └── test_tensor.cpp         # Unit tests
├── web_gui/
│   └── index.html              # Single-page web interface
└── third_party/
    ├── httplib.h               # cpp-httplib (header-only HTTP server)
    └── nlohmann/json.hpp       # nlohmann/json (header-only JSON)
```

## Web GUI

The web interface lets you:

1. **Configure models** — add/remove layers, set activations, choose optimizers
2. **Pick datasets** — XOR or Circle classification
3. **Train in real-time** — watch the loss curve update live
4. **View predictions** — see model outputs after training

You can also edit the config as raw JSON for full control.

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/schema` | Available layers, activations, optimizers, losses |
| `GET` | `/api/datasets` | Available datasets |
| `GET` | `/api/demo` | Run a quick XOR demo (synchronous) |
| `POST` | `/api/train` | Start a training job |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/:id` | Get job status + loss history |
| `POST` | `/api/jobs/:id/stop` | Stop a running job |
| `DELETE` | `/api/jobs/:id` | Delete a job |

### Example: Start a training job

```bash
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 2000,
    "learning_rate": 0.5,
    "optimizer": "sgd",
    "loss": "bce",
    "dataset": "xor",
    "layers": [
      {"type": "dense", "units": 8, "activation": "tanh"},
      {"type": "dense", "units": 1, "activation": "sigmoid"}
    ]
  }'
```

### Example: Check job status

```bash
curl http://localhost:8080/api/jobs/job_1_1234567890
```

Response:
```json
{
  "id": "job_1_1234567890",
  "status": "completed",
  "epochs": 2000,
  "current_epoch": 2000,
  "current_loss": 0.000658,
  "loss_history": [0.758, 0.512, 0.341, ...],
  "elapsed_ms": 45
}
```

## Using Neon as a Library

```cpp
#include "neon/neon.h"
using namespace neon;

int main() {
    // Create dataset
    Tensor X(Shape({4, 2}), {0,0, 0,1, 1,0, 1,1});
    Tensor Y(Shape({4, 1}), {0, 1, 1, 0});

    // Build model
    Sequential model;
    model.add(std::make_shared<Linear>(2, 8));
    model.add(std::make_shared<Tanh>());
    model.add(std::make_shared<Linear>(8, 1));
    model.add(std::make_shared<Sigmoid>());

    // Train
    auto optimizer = make_sgd(model, 0.5f, 0.9f);
    for (int epoch = 0; epoch < 5000; ++epoch) {
        Tensor pred = model.forward(X);
        float loss = loss::binary_cross_entropy(pred, Y);
        Tensor grad = loss::binary_cross_entropy_grad(pred, Y);
        model.backward(grad);
        optimizer->step();
        optimizer->zero_grad();
    }

    // Predict
    model.eval();
    Tensor pred = model.forward(X);
    std::cout << pred << std::endl;
}
```

## Docker Compose

```yaml
services:
  neon:
    build: .
    ports:
      - "8080:8080"
    restart: unless-stopped
```

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

## License

MIT
