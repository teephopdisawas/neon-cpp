#include "neon/server.h"
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <unistd.h>

neon::NeonServer* g_server = nullptr;

void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down...\n";
    if (g_server) g_server->stop();
}

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string web_root = "./web_gui";
    if (argc > 1) port = std::atoi(argv[1]);
    if (argc > 2) web_root = argv[2];

    // If web_gui doesn't exist relative to cwd, try relative to binary location
    if (web_root == "./web_gui" && access("./web_gui/index.html", F_OK) != 0) {
        // Try parent directory (running from build/)
        if (access("../web_gui/index.html", F_OK) == 0) {
            web_root = "../web_gui";
        }
    }

    std::cout << R"(
    ╔═══════════════════════════════════════╗
    ║   🔶 Neon AI Framework v0.2.0        ║
    ║   C++ Neural Network with Web GUI     ║
    ╚═══════════════════════════════════════╝
    )" << "\n";

    neon::NeonServer server(port, web_root);
    g_server = &server;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    server.start();
    return 0;
}
