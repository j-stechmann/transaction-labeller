# ---- deps: build dependency cache layer ----
FROM rust:1-slim-bookworm@sha256:1469a27c125cb5a3aebfa4f4e4665d935b02fb72cc093b2c974b3d740e43f157 AS deps
WORKDIR /build
# curl/unzip: utoipa-swagger-ui downloads the Swagger UI bundle at build time
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs
COPY Cargo.toml Cargo.lock ./
RUN cargo fetch --locked

# ---- build: compile the actual sources ----
FROM deps AS build
COPY src ./src
COPY tests ./tests
ARG SKIP_TESTS=""
RUN if [ -z "$SKIP_TESTS" ]; then cargo test --locked --all-features; else echo "SKIP_TESTS=1: skipping tests"; fi
RUN cargo build --release --locked

# ---- runtime: minimal image ----
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS runtime
LABEL org.opencontainers.image.source="https://github.com/j-stechmann/transaction-labeller"
LABEL org.opencontainers.image.description="Local LLM-based REST service that labels bank transactions with dynamically generated category names"
LABEL org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 1000 labeller \
    && useradd --system --uid 1000 --gid labeller --create-home labeller \
    && mkdir -p /data \
    && chown labeller:labeller /data

WORKDIR /app
COPY --from=build /build/target/release/transaction-labeller /usr/local/bin/transaction-labeller

USER labeller
VOLUME ["/data"]

ENV TL_BIND_ADDR="0.0.0.0:8080" \
    TL_LABEL_LIBRARY="/data/labels.json"
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/transaction-labeller"]