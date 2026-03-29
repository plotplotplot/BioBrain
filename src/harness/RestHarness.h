#pragma once

#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>
#include <netinet/in.h>

// Minimal single-threaded HTTP server for testing BioBrain simulation.
// No external dependencies — uses POSIX sockets directly.
class RestHarness {
public:
    using Handler = std::function<std::string(const std::string& path,
                                              const std::string& body)>;

    explicit RestHarness(int port = 8080);
    ~RestHarness();

    // Register a handler for a path prefix (e.g., "/api/regions")
    void route(const std::string& path, Handler handler);

    // Start listening (blocks until stop() is called from another thread)
    void start();
    void startAsync();
    void stop();
    bool isRunning() const { return running_.load(); }

private:
    int port_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::jthread server_thread_;

    std::map<std::string, Handler> routes_;
    std::mutex routes_mutex_;

    void acceptLoop();
    void handleClient(int client_fd);

    // Parse HTTP request, return {method, path, body}
    struct Request {
        std::string method;
        std::string path;
        std::string body;
    };
    Request parseRequest(const std::string& raw);

    // Send HTTP response
    void sendResponse(int client_fd, int status, const std::string& body,
                      const std::string& content_type = "application/json");
};
