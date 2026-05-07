# Multi-stage build for Neon AI Framework
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY . .

# Build
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ─── Runtime stage ───────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home neon

WORKDIR /app

# Copy built binary and web GUI
COPY --from=builder /build/build/neon_server /app/neon_server
COPY --from=builder /build/web_gui /app/web_gui

# Make executable
RUN chmod +x /app/neon_server && chown -R neon:neon /app

USER neon

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

ENTRYPOINT ["/app/neon_server"]
CMD ["8080"]
