# syntax=docker/dockerfile:1
# htdemucs-rs for Cloudflare Containers (linux/amd64): the unchanged separator CLI, the
# htdemucs-serve entrypoint (src/bin/htdemucs-serve.rs) and the converted weights, baked in.
# See docs/container.md.
#
#   docker build --platform linux/amd64 --build-context weights=weights -t htdemucs-rs .
#   docker build ... --target debian       # the same on debian:trixie-slim (has a shell)
#   docker build ... --target melroformer  # + weights/melroformer-f16.safetensors as melroformer
#
# `weights` is a named build context: the git-ignored weights/ directory made by
# tools/convert.py (it is not part of the main context, see .dockerignore).

# Builder and runtime share Debian 13 (glibc 2.41). glibc, not musl: the CLI's malloc tuning is
# glibc's mallopt, and musl's allocator is slow for this workload.
ARG RUST_IMAGE=rust:1.99-slim-trixie@sha256:f6c4a35d21dbb3c89d7a8d056c193e94fc09037f2b58dd858b09c39a47dd02d7
ARG DISTROLESS_IMAGE=gcr.io/distroless/cc-debian13:nonroot@sha256:532c7a94f9e96ba9db75eb9388f4227751131c7cb0a411d1a8521af4096135de
ARG DEBIAN_IMAGE=debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2

FROM ${RUST_IMAGE} AS build
WORKDIR /src
# Dependencies first (their own layer), then the crate. The release profile (fat LTO, one codegen
# unit, stripped) comes from Cargo.toml; no target-cpu flags: the GEMM picks AVX2 / AVX-512 at
# run time.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs && touch src/lib.rs \
 && cargo build --release --locked --bin htdemucs-rs \
 && rm -rf src
COPY src ./src
RUN touch src/main.rs src/lib.rs \
 && cargo build --release --locked --bin htdemucs-rs --bin htdemucs-serve \
 && mkdir /out && cp target/release/htdemucs-rs target/release/htdemucs-serve /out/

# Weights in their own layer, below the binaries: a code change re-ships only the ~5 MB binary
# layer, not the 139 MB of weights.
FROM scratch AS weights-demucs
COPY --from=weights htdemucs.json htdemucs.safetensors htdemucs_6s.json htdemucs_6s.safetensors /weights/

FROM ${DISTROLESS_IMAGE} AS distroless
COPY --from=weights-demucs /weights /weights
COPY --from=build /out/ /usr/local/bin/
ENV HTDEMUCS_WEIGHTS=/weights PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/htdemucs-serve"]

FROM ${DEBIAN_IMAGE} AS debian
COPY --from=weights-demucs /weights /weights
COPY --from=build /out/ /usr/local/bin/
ENV HTDEMUCS_WEIGHTS=/weights PORT=8080
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/htdemucs-serve"]

# Mel-Band RoFormer (`-n melroformer`) as one more layer on top of the default image.
FROM distroless AS melroformer
COPY --from=weights melroformer.json /weights/melroformer.json
COPY --from=weights melroformer-f16.safetensors /weights/melroformer.safetensors

# Default target.
FROM distroless
