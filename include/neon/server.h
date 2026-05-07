#pragma once

#include "neon/neon.h"
#include <nlohmann/json.hpp>
#include "httplib.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <sstream>
#include <iostream>
#include <chrono>
#include <map>
#include <random>

// We use nlohmann/json for JSON parsing. Header-only, fetch from GitHub.
// For now we include a minimal JSON subset or use manual parsing.
// Actually, let's use a simple approach — embed the single-header json.hpp.
#include <nlohmann/json.hpp>

namespace neon {

using json = nlohmann::json;

// ─── Training Job State ─────────────────────────────────────────
struct TrainingJob {
    std::string id;
    std::string status;        // "idle", "running", "completed", "error"
    std::string model_json;    // serialized model config
    int epochs = 0;
    int current_epoch = 0;
    float learning_rate = 0.01f;
    float current_loss = 0.0f;
    std::vector<float> loss_history;
    std::string error_message;
    std::chrono::steady_clock::time_point start_time;

    json to_json() const {
        json j;
        j["id"] = id;
        j["status"] = status;
        j["epochs"] = epochs;
        j["current_epoch"] = current_epoch;
        j["learning_rate"] = learning_rate;
        j["current_loss"] = current_loss;
        j["loss_history"] = loss_history;
        j["error_message"] = error_message;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        j["elapsed_ms"] = (status == "running" || status == "completed") ? ms : 0;
        return j;
    }
};

// ─── Model Config from JSON ─────────────────────────────────────
struct LayerConfig {
    std::string type;
    int units = 0;
    std::string activation;
    float dropout_rate = 0.0f;
};

struct ModelConfig {
    std::string name;
    std::vector<LayerConfig> layers;
    float learning_rate = 0.01f;
    int epochs = 100;
    std::string optimizer = "sgd";  // "sgd" or "adam"
    std::string loss = "mse";       // "mse", "bce", "cross_entropy"
    std::string dataset = "xor";    // "xor", "circle", "custom"
};

ModelConfig parse_model_config(const json& j) {
    ModelConfig config;
    config.name = j.value("name", "unnamed");
    config.learning_rate = j.value("learning_rate", 0.01f);
    config.epochs = j.value("epochs", 100);
    config.optimizer = j.value("optimizer", "sgd");
    config.loss = j.value("loss", "mse");
    config.dataset = j.value("dataset", "xor");

    if (j.contains("layers") && j["layers"].is_array()) {
        for (auto& lj : j["layers"]) {
            LayerConfig lc;
            lc.type = lj.value("type", "dense");
            lc.units = lj.value("units", 4);
            lc.activation = lj.value("activation", "relu");
            lc.dropout_rate = j.value("dropout_rate", 0.0f);
            config.layers.push_back(lc);
        }
    }
    return config;
}

// ─── Dataset Generation ─────────────────────────────────────────
struct Dataset {
    Tensor X, Y;
    std::string name;
};

Dataset make_xor_dataset() {
    Tensor X(Shape({4, 2}), {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor Y(Shape({4, 1}), {0, 1, 1, 0});
    return {X, Y, "xor"};
}

// Circle dataset: classify points inside/outside a circle
Dataset make_circle_dataset(size_t n_samples = 200, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    std::vector<float> x_data, y_data;
    x_data.reserve(n_samples * 2);
    y_data.reserve(n_samples * 1);

    for (int i = 0; i < n_samples; ++i) {
        float x = dist(gen);
        float y = dist(gen);
        x_data.push_back(x);
        x_data.push_back(y);
        // Inside circle of radius 1?
        float label = (x*x + y*y < 1.0f) ? 1.0f : 0.0f;
        y_data.push_back(label);
    }

    Tensor X(Shape({n_samples, 2}), std::move(x_data));
    Tensor Y(Shape({n_samples, 1}), std::move(y_data));
    return {X, Y, "circle"};
}

Dataset make_dataset(const std::string& name) {
    if (name == "circle") return make_circle_dataset();
    return make_xor_dataset();  // default
}

// ─── Build Model from Config ────────────────────────────────────
std::unique_ptr<Sequential> build_model(const ModelConfig& config, size_t input_dim) {
    auto model = std::make_unique<Sequential>();

    size_t in_dim = input_dim;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        const auto& lc = config.layers[i];
        bool is_last = (i == config.layers.size() - 1);

        // Add linear layer
        model->add(std::make_shared<Linear>(in_dim, lc.units));
        in_dim = lc.units;

        // Add activation (last layer might use sigmoid for binary classification)
        if (is_last) {
            if (lc.activation == "sigmoid" || config.loss == "bce") {
                model->add(std::make_shared<Sigmoid>());
            } else if (lc.activation == "softmax" || config.loss == "cross_entropy") {
                model->add(std::make_shared<Softmax>());
            } else {
                model->add(std::make_shared<Sigmoid>());
            }
        } else {
            if (lc.activation == "relu") {
                model->add(std::make_shared<ReLU>());
            } else if (lc.activation == "tanh") {
                model->add(std::make_shared<Tanh>());
            } else if (lc.activation == "sigmoid") {
                model->add(std::make_shared<Sigmoid>());
            } else {
                model->add(std::make_shared<ReLU>());
            }
            if (lc.dropout_rate > 0.0f) {
                model->add(std::make_shared<Dropout>(lc.dropout_rate));
            }
        }
    }

    return model;
}

// ─── Compute Loss ───────────────────────────────────────────────
float compute_loss(const std::string& loss_name, const Tensor& pred, const Tensor& target) {
    if (loss_name == "bce") return loss::binary_cross_entropy(pred, target);
    if (loss_name == "cross_entropy") return loss::cross_entropy(pred, target);
    return loss::mse(pred, target);
}

Tensor compute_loss_grad(const std::string& loss_name, const Tensor& pred, const Tensor& target) {
    if (loss_name == "bce") return loss::binary_cross_entropy_grad(pred, target);
    if (loss_name == "cross_entropy") return loss::cross_entropy_grad(pred, target);
    return loss::mse_grad(pred, target);
}

// ─── REST API Server ────────────────────────────────────────────
class NeonServer {
public:
    NeonServer(int port = 8080, const std::string& web_root = "./web_gui")
        : port_(port), web_root_(web_root), running_(false) {}

    void start() {
        running_ = true;

        // Serve static files from web_gui directory
        svr_.set_mount_point("/", web_root_);

        // ── API Endpoints ───────────────────────────────────────

        // Health check
        svr_.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
            json j = {{"status", "ok"}, {"framework", "neon-cpp"}, {"version", "0.2.0"}};
            res.set_content(j.dump(), "application/json");
        });

        // List available datasets
        svr_.Get("/api/datasets", [](const httplib::Request&, httplib::Response& res) {
            json j = {
                {"datasets", {
                    {{"name", "xor"}, {"description", "XOR classification (4 samples)"}, {"input_dim", 2}, {"output_dim", 1}},
                    {{"name", "circle"}, {"description", "Circle classification (200 samples)"}, {"input_dim", 2}, {"output_dim", 1}}
                }}
            };
            res.set_content(j.dump(), "application/json");
        });

        // List available layers, activations, optimizers, losses
        svr_.Get("/api/schema", [](const httplib::Request&, httplib::Response& res) {
            json j = {
                {"activations", {"relu", "sigmoid", "tanh", "softmax"}},
                {"optimizers", {"sgd", "adam"}},
                {"losses", {"mse", "bce", "cross_entropy"}},
                {"layer_types", {"dense"}},
                {"datasets", {"xor", "circle"}}
            };
            res.set_content(j.dump(), "application/json");
        });

        // List all jobs
        svr_.Get("/api/jobs", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mutex_);
            json jobs = json::array();
            for (auto& [id, job] : jobs_) {
                jobs.push_back(job.to_json());
            }
            res.set_content(jobs.dump(), "application/json");
        });

        // Get job status
        svr_.Get("/api/jobs/([a-zA-Z0-9_-]+)", [this](const httplib::Request& req, httplib::Response& res) {
            std::string job_id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                res.status = 404;
                res.set_content(json{{"error", "job not found"}}.dump(), "application/json");
                return;
            }
            res.set_content(it->second.to_json().dump(), "application/json");
        });

        // Create and start a training job
        svr_.Post("/api/train", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json body = json::parse(req.body);
                ModelConfig config = parse_model_config(body);

                // Generate job ID
                std::string job_id = generate_job_id();

                // Create job
                TrainingJob job;
                job.id = job_id;
                job.status = "running";
                job.model_json = body.dump();
                job.epochs = config.epochs;
                job.learning_rate = config.learning_rate;
                job.start_time = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    jobs_[job_id] = std::move(job);
                }

                // Start training in background thread
                std::thread([this, job_id, config]() {
                    run_training(job_id, config);
                }).detach();

                json response = {{"job_id", job_id}, {"status", "started"}};
                res.set_content(response.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });

        // Stop a training job
        svr_.Post("/api/jobs/([a-zA-Z0-9_-]+)/stop", [this](const httplib::Request& req, httplib::Response& res) {
            std::string job_id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                res.status = 404;
                res.set_content(json{{"error", "job not found"}}.dump(), "application/json");
                return;
            }
            it->second.status = "stopped";
            res.set_content(json{{"status", "stopped"}}.dump(), "application/json");
        });

        // Delete a job
        svr_.Delete("/api/jobs/([a-zA-Z0-9_-]+)", [this](const httplib::Request& req, httplib::Response& res) {
            std::string job_id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.erase(job_id);
            res.set_content(json{{"status", "deleted"}}.dump(), "application/json");
        });

        // Run a quick XOR demo (synchronous, for testing)
        svr_.Get("/api/demo", [](const httplib::Request&, httplib::Response& res) {
            auto dataset = make_xor_dataset();

            ModelConfig config;
            config.epochs = 2000;
            config.learning_rate = 0.5f;
            config.optimizer = "sgd";
            config.loss = "bce";
            config.layers = {
                {"dense", 8, "tanh"},
                {"dense", 1, "sigmoid"}
            };

            auto model = build_model(config, 2);
            auto optimizer = make_sgd(*model, config.learning_rate, 0.9f);

            std::vector<float> loss_history;
            loss_history.reserve(config.epochs / 10);

            for (int epoch = 0; epoch < config.epochs; ++epoch) {
                Tensor pred = model->forward(dataset.X);
                float loss_val = compute_loss(config.loss, pred, dataset.Y);
                Tensor grad = compute_loss_grad(config.loss, pred, dataset.Y);
                model->backward(grad);
                optimizer->step();
                optimizer->zero_grad();

                if (epoch % 10 == 0) loss_history.push_back(loss_val);
            }

            // Get predictions
            model->eval();
            Tensor pred = model->forward(dataset.X);

            json predictions = json::array();
            for (int i = 0; i < 4; ++i) {
                predictions.push_back({
                    {"input", {dataset.X(i, 0), dataset.X(i, 1)}},
                    {"predicted", pred(i, 0)},
                    {"expected", dataset.Y(i, 0)}
                });
            }

            json j = {
                {"loss_history", loss_history},
                {"predictions", predictions},
                {"epochs", config.epochs},
                {"final_loss", loss_history.back()}
            };
            res.set_content(j.dump(), "application/json");
        });

        std::cout << "Neon server starting on port " << port_ << "...\n";
        std::cout << "  API: http://localhost:" << port_ << "/api/\n";
        std::cout << "  GUI: http://localhost:" << port_ << "/\n";
        svr_.listen("0.0.0.0", port_);
    }

    void stop() {
        running_ = false;
        svr_.stop();
    }

private:
    httplib::Server svr_;
    int port_;
    std::string web_root_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::map<std::string, TrainingJob> jobs_;
    int job_counter_ = 0;

    std::string generate_job_id() {
        return "job_" + std::to_string(++job_counter_) + "_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void run_training(const std::string& job_id, const ModelConfig& config) {
        try {
            auto dataset = make_dataset(config.dataset);
            auto model = build_model(config, dataset.X.shape().dims[1]);

            std::unique_ptr<Optimizer> optimizer;
            if (config.optimizer == "adam") {
                optimizer = std::make_unique<Adam>(model->parameters(), model->gradients(), config.learning_rate);
            } else {
                optimizer = std::make_unique<SGD>(model->parameters(), model->gradients(), config.learning_rate, 0.9f);
            }

            for (int epoch = 0; epoch < config.epochs; ++epoch) {
                // Check if stopped
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = jobs_.find(job_id);
                    if (it == jobs_.end() || it->second.status == "stopped") return;
                }

                Tensor pred = model->forward(dataset.X);
                float loss_val = compute_loss(config.loss, pred, dataset.Y);
                Tensor grad = compute_loss_grad(config.loss, pred, dataset.Y);
                model->backward(grad);
                optimizer->step();
                optimizer->zero_grad();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = jobs_.find(job_id);
                    if (it != jobs_.end()) {
                        it->second.current_epoch = epoch + 1;
                        it->second.current_loss = loss_val;
                        if (epoch % 5 == 0 || epoch == config.epochs - 1) {
                            it->second.loss_history.push_back(loss_val);
                        }
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = jobs_.find(job_id);
                if (it != jobs_.end()) {
                    it->second.status = "completed";
                }
            }

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = jobs_.find(job_id);
            if (it != jobs_.end()) {
                it->second.status = "error";
                it->second.error_message = e.what();
            }
        }
    }
};

} // namespace neon
