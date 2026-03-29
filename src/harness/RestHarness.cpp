#include "harness/RestHarness.h"
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

RestHarness::RestHarness(int port) : port_(port) {}

RestHarness::~RestHarness() {
    stop();
}

void RestHarness::route(const std::string& path, Handler handler) {
    std::lock_guard lock(routes_mutex_);
    routes_[path] = std::move(handler);
}

void RestHarness::start() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "RestHarness: socket() failed\n";
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "RestHarness: bind() failed on port " << port_ << "\n";
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    listen(server_fd_, 16);
    running_ = true;

    std::cout << "RestHarness listening on http://localhost:" << port_ << "\n";
    acceptLoop();
}

void RestHarness::startAsync() {
    server_thread_ = std::jthread([this](std::stop_token) { start(); });
}

void RestHarness::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void RestHarness::acceptLoop() {
    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd_,
                               reinterpret_cast<sockaddr*>(&client_addr),
                               &client_len);
        if (client_fd < 0) {
            if (running_) std::cerr << "RestHarness: accept() failed\n";
            continue;
        }

        handleClient(client_fd);
        close(client_fd);
    }
}

void RestHarness::handleClient(int client_fd) {
    char buf[65536];
    ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return;
    buf[n] = '\0';

    Request req = parseRequest(std::string(buf, n));

    // CORS headers for browser access
    std::string cors = "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: Content-Type\r\n";

    if (req.method == "OPTIONS") {
        std::string resp = "HTTP/1.1 204 No Content\r\n" + cors + "\r\n";
        send(client_fd, resp.c_str(), resp.size(), 0);
        return;
    }

    // Find matching route (longest prefix match)
    Handler handler;
    {
        std::lock_guard lock(routes_mutex_);
        std::string best_match;
        for (auto& [prefix, h] : routes_) {
            if (req.path.starts_with(prefix) && prefix.size() > best_match.size()) {
                best_match = prefix;
                handler = h;
            }
        }
    }

    std::string body;
    int status = 200;
    if (handler) {
        try {
            body = handler(req.path, req.body);
        } catch (const std::exception& e) {
            status = 500;
            body = std::string("{\"error\":\"") + e.what() + "\"}";
        }
    } else {
        status = 404;
        body = "{\"error\":\"Not found\",\"path\":\"" + req.path + "\"}";
    }

    sendResponse(client_fd, status, body);
}

RestHarness::Request RestHarness::parseRequest(const std::string& raw) {
    Request req;
    std::istringstream stream(raw);
    stream >> req.method >> req.path;

    // Find body after \r\n\r\n
    auto pos = raw.find("\r\n\r\n");
    if (pos != std::string::npos) {
        req.body = raw.substr(pos + 4);
    }

    return req;
}

void RestHarness::sendResponse(int client_fd, int status, const std::string& body,
                                const std::string& content_type) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        default: status_text = "Unknown"; break;
    }

    std::ostringstream resp;
    resp << "HTTP/1.1 " << status << " " << status_text << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Connection: close\r\n"
         << "\r\n"
         << body;

    std::string r = resp.str();
    send(client_fd, r.c_str(), r.size(), 0);
}
