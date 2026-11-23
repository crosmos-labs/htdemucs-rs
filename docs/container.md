# htdemucs-rs on Cloudflare Containers

The separator packaged as one linux/amd64 image for
[Cloudflare Containers](https://developers.cloudflare.com/containers/) (Workers Paid plan), with a
small HTTP entrypoint that a Durable Object can drive. It is built and tested on GitHub Actions
(`.github/workflows/container.yml`). Nothing here deploys anything or pushes an image anywhere.

Files: `Dockerfile`, `.dockerignore`, `src/bin/htdemucs-serve.rs` (the entrypoint),
`.github/workflows/container.yml`, and `.github/container/` (the workflow's job driver, parity check
and size script). The separator CLI (`src/main.rs`) is unchanged; the entrypoint runs it as a
child process.

## Decisions

| | chosen | why |
|---|---|---|
| runtime base | `gcr.io/distroless/cc-debian13:nonroot` | 173 MB uncompressed and 141 MB compressed in all, against 225 MB / 161 MB on `debian:trixie-slim`. Same glibc. No shell, runs as a non-root user. |
| C library | glibc (Debian 13, 2.41) | The CLI's malloc tuning (`HTDEMUCS_MALLOC_TUNE`, `src/model.rs`) is glibc's `mallopt`, and musl's allocator is slow for this workload. |
| weights | baked into the image, in their own layer | 139 MB is 0.7–7% of the image limit. Cold starts need no download and no network access. A code change re-ships only the 4.9 MB binary layer. |
| entrypoint | `htdemucs-serve`: an HTTP job server on :8080 | A song can take longer than any single request should be held open, so the interface is async: POST the audio, poll the status, GET a tar of the stems. One job per container. |
| instance | `standard-4` (4 vCPU, 12 GiB, 20 GB), one job per instance | It gives the shortest wall time, which is what memory is billed on, so it is the cheapest per song. Memory no longer decides it: since the streamed output (`a7c5ba8`), a 22-minute ensemble peaks at 1.5 GiB (7.0 GiB before), which every type from `standard-1` up holds. See *Sizing*. |

## Image layout

`Dockerfile` stages:

1. `build`: `rust:1.99-slim-trixie`. It builds the dependencies first, in a cached layer, then
   `htdemucs-rs` and `htdemucs-serve` with the crate's release profile (fat LTO, one codegen
   unit, stripped). There are no `target-cpu` flags: the GEMM picks AVX2 or AVX-512 at run time.
2. `weights-demucs`: `htdemucs.{json,safetensors}` and `htdemucs_6s.{json,safetensors}`, copied
   from the named build context `weights` (the git-ignored `weights/` that `tools/convert.py`
   makes).
3. `distroless` (the default target) and `debian`: base, then `/weights`, then
   `/usr/local/bin/{htdemucs-rs,htdemucs-serve}`. They set `HTDEMUCS_WEIGHTS=/weights` and
   `PORT=8080`, run as uid 65532, and use `ENTRYPOINT ["/usr/local/bin/htdemucs-serve"]`.
4. `melroformer`: the default image plus `weights/melroformer.json` and
   `weights/melroformer-f16.safetensors`, installed as `/weights/melroformer.safetensors`. This
   target is not built in CI.

Base images are pinned by digest. To update one, change the `ARG` at the top of the Dockerfile.

### Sizes

Measured on the runner by `.github/container/sizes.py`, which reads `docker save` layer by layer.
MB = 10^6 bytes. "Compressed" is gzip -6 per layer, which is what a registry stores and an
instance pulls.

| image | layers | uncompressed | compressed |
|---|---|---|---|
| **htdemucs-rs:distroless** | 23 | **173.0 MB** | **141.4 MB** |
| htdemucs-rs:debian | 3 | 225.1 MB | 160.8 MB |
| gcr.io/distroless/cc-debian13:nonroot (base) | 21 | 29.1 MB | 10.7 MB |
| debian:trixie-slim (base) | 1 | 81.3 MB | 30.0 MB |

Layers of the default image:

| layer | uncompressed | compressed |
|---|---|---|
| distroless/cc base (glibc, libgcc, libstdc++, libssl, tzdata, CA certificates) | 29.1 MB | 10.7 MB |
| `/weights` (htdemucs 84.0 MB + htdemucs_6s 54.9 MB, fp16) | 138.9 MB | 128.7 MB |
| `/usr/local/bin` (`htdemucs-rs` 4.35 MB, `htdemucs-serve` 0.55 MB) | 4.9 MB | 2.0 MB |

The fp16 weights barely compress (7%), so the weights dominate the image whichever base you use.
The debian-slim variant costs 52 MB more on disk and 19 MB more to pull, and buys a shell for
debugging (`docker run --entrypoint sh`). Inside the image the two run the same binary; the CI
smoke run checks both.

With Mel-RoFormer as a third model, the image grows by the size of its weights file:
`weights/melroformer-f16.safetensors` is **456 MB**, not the 100–200 MB assumed, and fp16 barely
compresses. The `melroformer` image would be about 630 MB uncompressed and 560 MB compressed. That
is still well under every instance type's image limit (below).

### Weights: baked in, not fetched at start

Cloudflare's limits
([Limits](https://developers.cloudflare.com/containers/platform-details/limits/), updated
2026-11-12):

- Image size is "Same as instance disk space": 2 GB on `lite` up to 20 GB on `standard-4`.
- Total image storage per account is 50 GB.
- A container's disk "is ephemeral by default, so each restart starts from a fresh copy of the
  image" ([Platform details](https://developers.cloudflare.com/containers/platform-details/)).
- Cold starts are "typically about 1 to 3 seconds, depending on image size and how long the
  entrypoint takes to run". Images "are pre-fetched to certain locations", and you pay only for
  running instances, not for prepared images.

| | baked (chosen) | fetched from R2 at start |
|---|---|---|
| image | 141 MB compressed | 13 MB compressed |
| every cold start | nothing extra; the image is pre-fetched | download 139 MB (on an ephemeral disk, so on *every* start), billed as memory and disk time while it runs |
| needs | nothing | `enableInternet` plus a URL to sign, or a FUSE mount of R2 (needs a shell and a FUSE binary, so not distroless; the docs call its performance "well below SSD-level") |
| deploy of a code change | 4.9 MB layer (the weights layer is unchanged) | 4.9 MB layer |
| failure modes | none new | R2 or network errors at start, and a version skew between binary and weights |

The image limit (2–20 GB) and the 50 GB account storage are far from 141 MB, or 630 MB with
Mel-RoFormer. Fetching would save only image storage, which is not a constraint, and it adds a
download to every cold start. So the weights are baked in.

The entrypoint is ready 0.2–0.7 s after `docker run` on the runner (`ready_s` in the results).
There is no model warm-up at start: each job loads its weights (139 MB of fp16 from local disk) in
its own process.

## Building and running

```sh
# from the repository root; weights/ from tools/convert.py (see README)
docker build --platform linux/amd64 --build-context weights=weights -t htdemucs-rs .
docker build --platform linux/amd64 --build-context weights=weights --target debian -t htdemucs-rs:debian .
docker build --platform linux/amd64 --build-context weights=weights --target melroformer -t htdemucs-rs:mr .

# HTTP server
docker run --rm -p 8080:8080 htdemucs-rs
curl -s localhost:8080/health
curl -s -X POST 'localhost:8080/job?name=ensemble&shifts=2' --data-binary @song.wav
curl -s localhost:8080/job                       # poll until "state":"done"
curl -s localhost:8080/job/result -o stems.tar   # htdemucs/{drums,bass,other,vocals}.flac, htdemucs_6s/...

# one job, no server: audio on stdin, tar on stdout, a JSON summary on stderr
docker run --rm -i htdemucs-rs once name=htdemucs shifts=2 < song.wav > stems.tar
```

If `weights/` is a symlink, as in a git worktree, pass its real path:
`--build-context weights=$(realpath weights)`. Without `--build-context weights=...` the build fails
at `COPY --from=weights`, because BuildKit then looks for an image called `weights`.

On Apple silicon the `--platform linux/amd64` build runs under emulation and is slow, so build on
an x86 machine or on Actions. A full build takes 3.3 min on a 2-vCPU runner, and 1–2 min when the
dependency layer is cached.

## The interface (`htdemucs-serve`)

The server uses only the standard library and libc, in about 750 lines. It listens on `$PORT`
(default 8080) and runs one job at a time.

| request | response |
|---|---|
| `GET /health` | `200 ok`: the readiness check |
| `POST /job?<options>`, body = the audio file | `202` + status JSON. `409` while a job is receiving or running, `400` on a bad option, `411` without a body length, `413` above `HTDEMUCS_MAX_UPLOAD_MB` (default 2048). The body can be sent with `Content-Length` or chunked, and `Expect: 100-continue` is honoured. |
| `GET /job` | status JSON (below) |
| `GET /job/result` | `200` with an uncompressed tar of the stems, named `<output folder>/<stem>.<ext>`, and an exact `Content-Length` (R2 can store the stream directly). `409` unless the job is done. |
| `DELETE /job` | Kills a running job (SIGKILL), or deletes a finished job's files. |

**Options** (query string, or `key=value` arguments in `once` mode) map one-to-one onto the CLI's
flags: `name` (or `n`), `shifts`, `seed`, `overlap`, `num-overlap`, `clip-mode`, `clip-mode-4`,
`ensemble-4`, `threads`, `ensemble-own-6s=1`, and `format` = `flac` (default) | `wav` | `float32`.

- Values must be plain tokens, so a request cannot reach `--weights`, `-o` or other flags.
- The CLI validates the values themselves; if it rejects one, the job fails with the CLI's
  message.
- The input format is sniffed from the first bytes (WAV, FLAC, MP4/M4A, MP3) to give the decoder
  its hint.

**Status JSON**, e.g. at the end of the 60 s ensemble sizing job:

```json
{"state":"done","id":1,"args":["-n","ensemble","--shifts","2","--flac"],"input_bytes":21168058,
 "progress":{"done":...,"total":...},"wall_s":177.71,"cpu_s":340.88,"peak_rss_mb":1055,"error":"",
 "files":[{"name":"htdemucs/bass.flac","bytes":...}, ...],
 "cpu_model":"AMD EPYC 9V74 80-Core Processor","nproc":2}
```

- `state` is `idle`, `receiving`, `running`, `done` or `failed`.
- `progress` is in seconds of audio and counts every shift pass and both ensemble models. It is
  parsed from the CLI's tqdm line, so it only moves forward.
- `cpu_s` and `peak_rss_mb` come from `wait4` on the job's process.
- `cpu_model` and `nproc` report the host, so the first deployment will show which CPU Cloudflare
  actually gives us.

**Design choices.**

- **One process per job.** The CLI's malloc tuning keeps freed memory in the heap, so a
  long-lived separator would sit at its peak between jobs. A child process hands its memory back
  when it ends. A crash in a job cannot take the server down, and the CLI stays the single
  source of truth.
- **SIGTERM and SIGINT kill the job and exit at once.** As PID 1 the server would otherwise
  ignore SIGTERM, and Cloudflare "waits up to 15 minutes" before SIGKILL, billing all the while.
- **`once` mode.** `htdemucs-serve once` fits `docker run -i`, and Cloudflare's
  `ctx.container.exec()` with stdin and stdout streams, for callers that would rather not use
  HTTP.

### Driving it from a Worker (sketch, not deployed)

Cloudflare's current API is the Durable Object Container API (`this.ctx.container`).
The `@cloudflare/containers` `Container` class "is deprecated"
([migration guide](https://developers.cloudflare.com/containers/guides/migrate-to-durable-object-container-api/)).
Its `defaultPort` and `containerFetch()` map to `getTcpPort(8080).fetch()`, and `sleepAfter` maps
to `setInactivityTimeout()` (maximum 6 h). `start()` "returns before the container is ready", so
poll `/health`
([API](https://developers.cloudflare.com/containers/api/durable-object-container/)).

```ts
import { DurableObject } from "cloudflare:workers";

export class Separator extends DurableObject<Env> {
  port() { return this.ctx.container!.getTcpPort(8080); }

  async ensureReady() {
    const c = this.ctx.container!;
    if (!c.running) c.start({ enableInternet: false, instance: "standard-4" });
    for (let i = 0; i < 300; i++) {
      try { if ((await this.port().fetch("http://c/health")).ok) return; } catch {}
      await scheduler.wait(100);
    }
    throw new Error("separator did not start");
  }

  // From the queue consumer: an R2 key of the decoded audio, and e.g. "name=ensemble&shifts=2".
  async submit(key: string, options: string) {
    await this.ensureReady();
    await this.ctx.container!.setInactivityTimeout(30 * 60_000);
    const audio = (await this.env.AUDIO.get(key))!;
    const r = await this.port().fetch(`http://c/job?${options}`, {
      method: "POST", body: audio.body, headers: { "content-length": String(audio.size) },
    });
    if (r.status !== 202) throw new Error(await r.text());
    await this.ctx.storage.put("key", key);
    await this.ctx.storage.setAlarm(Date.now() + 15_000);
  }

  async alarm() {
    const s: any = await (await this.port().fetch("http://c/job")).json();
    if (s.state === "running") return this.ctx.storage.setAlarm(Date.now() + 15_000);
    const key = await this.ctx.storage.get<string>("key");
    if (s.state === "done") {
      const tar = await this.port().fetch("http://c/job/result");   // has Content-Length
      await this.env.STEMS.put(`${key}.stems.tar`, tar.body);
    }
    this.ctx.container!.signal(15);   // stop now: memory and disk are billed while it runs
  }
}
```

Concurrency is a pool of Durable Object names (`getByName("sep-0" ... "sep-N")`) with one job
each. `max_instances` caps the pool.

## Tested on GitHub Actions

`.github/workflows/container.yml` runs on a push to `container`, or by hand with inputs `sizing`,
`threads_1` and `long_minutes`.

1. It builds both images with buildx and the GHA layer cache, and reports their sizes.
2. It makes a 12 s synthetic clip (`.github/bench/synth.py`, no copyrighted audio) and PyTorch
   references on the runner (`tools/ref_separate.py`: demucs 4.0.1, torch 2.2.2 CPU, shifts 2,
   seed 0).
3. Each smoke and sizing job runs in a fresh container through the HTTP interface
   (`.github/container/job.sh`). That makes the cgroup's `memory.peak` the job's own.
4. The weights come from the same Actions cache as `bench.yml`, or are converted on the runner.

**Parity** (container output vs PyTorch on the same runner; SNR in dB per stem; float32 output,
`--clip-mode none`):

| | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|
| htdemucs | 116.6 | 117.5 | 119.3 | 88.6 (n-s) | | |
| htdemucs_6s | 125.1 | 128.8 | 126.7 | 90.9 (n-s) | 95.0 (n-s) | 93.6 (n-s) |
| ensemble, 4-stem | 121.9 | 122.4 | 124.4 | 92.0 (n-s) | | |
| ensemble, 6-stem | 121.9 | 122.4 | 124.4 | 92.0 (n-s) | 95.0 (n-s) | 93.6 (n-s) |
| htdemucs_6s own stems from the ensemble run (`ensemble-own-6s=1`) | 123.9 | 117.4 | 119.4 | 90.7 (n-s) | 94.7 (n-s) | 93.2 (n-s) |

The first four rows are from run 2. The last row is from run 3, on another EPYC 9V74 host whose
figures for the other rows came out 1–11 dB different (htdemucs 114.5 / 116.5 / 118.1; 6s
123.9 / 117.4 / 119.4). Output is deterministic per kernel path, not across hosts (README,
*Determinism*). Every audible stem in every run is at or above 114 dB.

- The ensemble reference is built from the two models' references, the way `src/ensemble.rs`
  combines them.
- "n-s" marks a near-silent stem: the synthetic clip has no real voice, guitar or piano, so these
  stems sit at -71 to -74 dBFS. SNR is relative to a stem's own level, so for these stems
  `check.py` requires the absolute error to be at or below -130 dBFS (90 dB under a -40 dBFS
  stem). It is -160 to -166 dBFS here.
- htdemucs' vocals give the same 88.6 dB in `bench.yml` on bare metal, with the binary built
  outside Docker. The whole htdemucs row matches that run to 0.1 dB: the container changes
  nothing.

**Runner CPUs seen.** GitHub assigns the CPU per run, always 2 vCPUs (one core with SMT):

- run 1: Intel Xeon Platinum 8573C;
- runs 2 and 3: AMD EPYC 9V74 (Zen 4, AVX-512).

All sizing numbers below are from the EPYC 9V74.

**Jobs** (EPYC 9V74, 2 vCPUs, the CLI's default of all cores = 2 threads; run 2 unless marked):

| job | wall s | CPU s | CPU s per audio s | peak RSS (separator) | cgroup memory.peak |
|---|---|---|---|---|---|
| htdemucs, shifts 2, 12 s | 26.2 | 50.4 | 4.20 | 628 MiB | 647 MiB |
| htdemucs_6s, shifts 2, 12 s | 21.7 | 41.7 | 3.47 | 667 MiB | 694 MiB |
| ensemble, shifts 2, 12 s | 48.0 | 91.2 | 7.60 | 802 MiB | 845 MiB |
| **ensemble, shifts 2, 60 s** (both models at shifts 2) | 177.7 | 340.9 | **5.68** | 1055 MiB | 1099 MiB |
| **ensemble, shifts 0, 60 s** | 90.9 | 174.3 | **2.91** | 1056 MiB | 1084 MiB |
| **ensemble, shifts 0, 240 s** | 349.5 | 666.9 | **2.78** | **1878 MiB** | 1976 MiB |
| htdemucs_6s, shifts 0, 240 s | 158.8 | 303.9 | 1.27 | 1418 MiB | 1546 MiB |
| ensemble, shifts 2, 60 s, `threads=1` (run 3) | 251.3 | 255.3 | **4.25** | 1016 MiB | 1060 MiB |
| **ensemble, shifts 0, 22 min** (run 3) | 2447.9 | 4547.7 | 3.45 | **7131 MiB** | 7193 MiB |

- The 12 s figures carry a fixed per-segment overhead: a 12 s clip still runs whole 7.8 s
  segments. The 60 s and 240 s clips are the 12 s clip tiled, and give the per-song rate.
- The cgroup peak includes the page cache of the uploaded input and the written stems. It is
  20–130 MiB above the separator's RSS.
- Run 3's host was slower than run 2's, on the same CPU model: the identical 12 s smoke jobs took
  22–28% more CPU there. So its CPU figures (`threads=1`, 22 min) are high by about that much
  against the rest of the table. Corrected for that, the 22-minute rate (3.45 / 1.28 = 2.7) agrees
  with the 240 s one (2.78).
- **One thread against two.** On the runner's two vCPUs (one core with SMT), two threads use
  about 70% more CPU-seconds than one thread, for only about 10% more throughput.
  - CPU: 5.68 CPU-s per audio s at 2 threads, against 4.25 / 1.28 = 3.3 at 1 thread,
    host-corrected.
  - Wall: 178 s at 2 threads, against about 251 / 1.28 = 196 s at 1 thread.
  - Billed CPU therefore depends on whether a Cloudflare vCPU is an SMT sibling or a whole core
    (open question 1). If it is a sibling, `threads` = the number of physical cores is the
    cheap setting.
- **Memory grows linearly with length**, about 4.6–5 MiB per second of audio for the ensemble (10
  stems of float32 output plus the decoded input), on top of about 0.8 GiB:

  | ensemble length | peak RSS |
  |---|---|
  | 60 s | 1055 MiB |
  | 4 min | 1878 MiB |
  | 22 min | 7131 MiB (7.0 GiB) |

  Shifts add time, not memory (60 s: 1055 MiB at shifts 2, 1056 at shifts 0). The 22-minute
  figure is above the 5.3 GB measured on macOS (README, footprint). One plausible cause is the
  glibc tuning that keeps freed memory in the heap; this was not measured.
- **Since the streamed output (`a7c5ba8`) memory hardly grows with length.** The stems go to the
  writers as the overlap-add window passes them, so only the decoded input (0.35 MB per second)
  grows. Peak RSS on bare-metal Linux (i5-12450H, 8 threads, the same glibc tuning; not yet
  measured in the container), the build before against `66c76a4`, all stems bit-identical:

  | | Aerials (4:04), shifts 2 | 22 min, shifts 0 |
  |---|---|---|
  | htdemucs | 1301 -> 886 MiB | 4036 -> 1213 MiB |
  | htdemucs_6s | 1491 -> 965 MiB | 4849 -> 1315 MiB |
  | ensemble (Fretboard's options) | 1994 -> 1271 MiB | **7203 -> 1561 MiB** |

  The "before" 22-minute ensemble (7203 MiB) matches the container's 7131 MiB. **In the
  container** (Blacksmith 4 vCPU AMD EPYC, run 45279954503, `main` at `9ce5c33`, distroless, the
  same 22-minute clip, `-n ensemble --shifts 0 --flac`): **peak RSS 1388 MiB** (was 7131), 589 s
  wall, 1743 CPU s (1.32 CPU s per audio s at 4 threads on Zen 5). Its cgroup `memory.peak` was
  6694 MiB, but that is page cache: the job ran at the default `--clip-mode rescale`, so all 10
  stems went through float temp files (4.7 GB written and read back), plus the 466 MB input and
  the 0.6 GB result and tar. Page cache is reclaimed under the cgroup's limit rather than
  causing an OOM; Fretboard's options rescale only the 4-stem view (1.9 GB of temp files at 22
  minutes) and `clamp` needs none. `--clip-mode rescale` stems
  also need temporary disk: 4 bytes x 2 channels per sample per rescaled stem (Fretboard's 4-stem
  view: 4 x 466 MB at 22 min), well inside every type's disk.

## Sizing

### Cloudflare's instance types and prices

From [Pricing](https://developers.cloudflare.com/containers/pricing/) and
[Limits](https://developers.cloudflare.com/containers/platform-details/limits/), read 2026-11-23:

| type | vCPU | memory | disk (= image limit) |
|---|---|---|---|
| lite | 1/16 | 256 MiB | 2 GB |
| basic | 1/4 | 1 GiB | 4 GB |
| standard-1 | 1/2 | 4 GiB | 8 GB |
| standard-2 | 1 | 6 GiB | 12 GB |
| standard-3 | 2 | 8 GiB | 16 GB |
| standard-4 | 4 | 12 GiB | 20 GB |

- **Custom types** `{vcpu, memory_mib, disk_mb}` are allowed. The per-instance maximum is 4 vCPU,
  12 GiB and 20 GB unless raised.
- **Included with Workers Paid**, per month: 375 vCPU-minutes, 25 GiB-hours of memory and
  200 GB-hours of disk.
- **Beyond that:** $0.000020 per vCPU-second ($0.072 per vCPU-hour), $0.0000025 per GiB-second
  ($0.009 per GiB-hour) and $0.00000007 per GB-second.
- **What is billed when.** CPU is billed on "active usage only". Memory and disk are billed on the
  "provisioned resources for the instance type you select" while the instance runs, from start
  until it sleeps.

So CPU cost depends on the work, while memory and disk cost depend on instance size times wall
time. Since every instance holds the whole 12 GiB of a `standard-4` until it stops, stopping the
container as soon as the result is fetched matters.

### Per-song cost model

- **CPU** = (CPU seconds per audio second) x song length. Measured: **5.43** for Fretboard's
  recipe (htdemucs + htdemucs_6s at shifts 2, i.e. `-n ensemble --shifts 2`) and **2.78** for the
  ensemble at shifts 0, both at 2 threads on the EPYC 9V74 (Zen 4, AVX-512). The recipe figure
  scales the 240 s shifts-0 run by the measured 60 s shifts-2 / shifts-0 ratio of 1.955. Running
  the two models as separate processes costs the same CPU and less memory.
- **Wall** = CPU / (vCPUs x efficiency) + 30 s. Efficiency is 0.9 at 2 vCPUs (assumed) and 0.78
  at 4, measured on Blacksmith's 8-vCPU AMD EPYC (Zen 5, AVX-512; bench run 45271063897):
  htdemucs takes 0.646 s per 7.8 s segment at 4 threads against 2.014 s at 1 thread, a 3.1x
  speed-up (8 threads: 0.537 s). The 30 s covers start, upload, result download and stop.
- **Memory** = the instance's memory x wall. **Disk** = the instance's disk x wall (1–6% of the
  inclusion per song, never the binding limit).
- **Songs inside the inclusion** = the smallest of 375 vCPU-min, 25 GiB-h and 200 GB-h, each
  divided by one song's use of it. **Cost per extra song** = CPU + memory + disk at the overage
  rates.

Per song, at the measured 2-thread rates (EPYC 9V74). The memory needs used for "fits?" are
those of the streamed output, about 1.3 GiB (4 min) and 1.6 GiB (22 min) plus page cache, so
every type from `standard-1` up holds every song. (Before `a7c5ba8` they were 1.83 and 6.96 GiB:
22-minute songs did not fit `standard-2` and only just fit `standard-3`.)

| recipe | song | instance | fits? | CPU / song | wall / song | memory / song | songs in the $5 inclusion (binding) | cost per extra song |
|---|---|---|---|---|---|---|---|---|
| Fretboard recipe: both models, shifts 2 | 4 min | standard-1 | yes | 21.7 vCPU-min | 43.9 min | 2.93 GiB-h | 8.5 (memory) | $0.054 |
| Fretboard recipe: both models, shifts 2 | 4 min | standard-2 | yes | 21.7 vCPU-min | 22.2 min | 2.22 GiB-h | 11.3 (memory) | $0.047 |
| Fretboard recipe: both models, shifts 2 | 4 min | standard-3 | yes | 21.7 vCPU-min | 12.6 min | 1.68 GiB-h | 14.9 (memory) | $0.042 |
| Fretboard recipe: both models, shifts 2 | 4 min | standard-4 | yes | 21.7 vCPU-min | 7.5 min | 1.49 GiB-h | 16.8 (memory) | $0.040 |
| Fretboard recipe: both models, shifts 2 | 22 min | standard-1 | yes | 119.5 vCPU-min | 239.5 min | 15.97 GiB-h | 1.6 (memory) | $0.295 |
| Fretboard recipe: both models, shifts 2 | 22 min | standard-2 | yes | 119.5 vCPU-min | 120.0 min | 12.00 GiB-h | 2.1 (memory) | $0.257 |
| Fretboard recipe: both models, shifts 2 | 22 min | standard-3 | yes | 119.5 vCPU-min | 66.9 min | 8.92 GiB-h | 2.8 (memory) | $0.228 |
| Fretboard recipe: both models, shifts 2 | 22 min | standard-4 | yes | 119.5 vCPU-min | 38.8 min | 7.76 GiB-h | 3.1 (CPU) | $0.217 |
| Ensemble, shifts 0 | 4 min | standard-1 | yes | 11.1 vCPU-min | 22.7 min | 1.52 GiB-h | 16.5 (memory) | $0.028 |
| Ensemble, shifts 0 | 4 min | standard-2 | yes | 11.1 vCPU-min | 11.6 min | 1.16 GiB-h | 21.5 (memory) | $0.024 |
| Ensemble, shifts 0 | 4 min | standard-3 | yes | 11.1 vCPU-min | 6.7 min | 0.89 GiB-h | 28.1 (memory) | $0.022 |
| Ensemble, shifts 0 | 4 min | standard-4 | yes | 11.1 vCPU-min | 4.1 min | 0.81 GiB-h | 30.8 (memory) | $0.021 |
| Ensemble, shifts 0 | 22 min | standard-1 | yes | 61.2 vCPU-min | 122.9 min | 8.19 GiB-h | 3.1 (memory) | $0.151 |
| Ensemble, shifts 0 | 22 min | standard-2 | yes | 61.2 vCPU-min | 61.7 min | 6.17 GiB-h | 4.1 (memory) | $0.132 |
| Ensemble, shifts 0 | 22 min | standard-3 | yes | 61.2 vCPU-min | 34.5 min | 4.60 GiB-h | 5.4 (memory) | $0.117 |
| Ensemble, shifts 0 | 22 min | standard-4 | yes | 61.2 vCPU-min | 20.1 min | 4.02 GiB-h | 6.1 (CPU) | $0.111 |

If a Cloudflare vCPU turns out to be a whole core (the `threads=1` rate), CPU per song drops by
roughly 40%. A 4-minute recipe song would then take about 13 vCPU-minutes (about 28 songs a month
inside the inclusion), and a shifts-0 ensemble song about 7 vCPU-minutes (about 50).

### Recommendation

- **Instance: `standard-4`** (4 vCPU, 12 GiB, 20 GB), **one job per instance**, a pool of 1–2
  Durable Object names, and `max_instances: 2`.
  - Memory is billed on wall time, so the fastest instance is also the cheapest per song. Every
    type now holds a 22-minute song (about 1.6 GiB since the streamed output, against 7.0 GiB
    before), so this is a cost choice, not a memory one. At the measured 0.78 efficiency it is
    still the cheapest, by a smaller margin than the assumed 0.9 gave ($0.040 against $0.042 on
    `standard-3` for a 4-minute recipe song). Inside the inclusion, memory and CPU run out at
    about the same point on `standard-4`; on the smaller types memory runs out first.
  - One job per instance because the work is CPU-bound. Two jobs in one container would only
    share the 4 vCPUs and add their memory.
  - The DO should stop the container as soon as it has the result (`signal(15)`, or `destroy()`),
    not wait for an inactivity timeout: every idle minute bills 12 GiB.
- **Fretboard's recipe** (htdemucs + htdemucs_6s at shifts 2), songs up to 22 minutes:
  - about **17 four-minute songs a month** inside the $5 inclusion (memory and CPU about equal:
    1.49 GiB-h and 21.7 vCPU-min each);
  - about **$0.040 per extra 4-minute song**, $0.22 per 22-minute one;
  - about 7.5 minutes of wall time for 4 minutes of music, at the measured 78% scaling to 4
    vCPUs.
- **Ensemble at shifts 0** (research round 2's "avg_s0": above htdemucs at shifts 2 on every
  MUSDB stem):
  - about **31 four-minute songs a month** inside the inclusion;
  - about **$0.021 per extra song** ($0.11 at 22 minutes);
  - about 4.1 minutes of wall time.
- **Cheaper still, if needed:**
  - A custom type with 4 vCPUs and less memory: with the streamed output,
    `{vcpu: 4, memoryMib: 3072}` holds any song up to 22 minutes (separator peak 1.4 GiB in the
    container), and cuts the memory part of a 4-minute recipe song from $0.013 to $0.003. CPU,
    the bigger part, stays. The page cache of the rescale temp files is the one thing to watch
    (open question 4).
  - Measure `--threads` against Cloudflare's real vCPUs first (open question 1).

## Open questions

1. **Which CPU Cloudflare gives us, and what a "vCPU" is.** The cost per song is the CPU rate,
   and it varies a lot by host. Per segment forward at 1 thread, the README measures 0.33 CPU-s
   per audio second on Zen 5 against 0.72 on Zen 3. On a 2-vCPU runner, 2 threads on one SMT
   core cost more CPU-seconds than 1 thread (see the `threads=1` row), and Cloudflare's docs say
   nothing about the CPU model or SMT. The status JSON reports `cpu_model` and `nproc`, so the
   first real job answers the model question. Run the same 60 s clip there before committing to
   a budget.
2. **Multi-core scaling on x86**, roughly answered: 3.1x at 4 threads (0.78 efficiency) and
   3.75x at 8 on Blacksmith's 8-vCPU Zen 5 (bench run 45271063897, htdemucs per segment). Those
   are 8 vCPUs used 4 at a time, not a 4-vCPU instance; Cloudflare's own scaling is still open
   (with question 1). It sets wall time, and so memory cost, not CPU cost.
3. **Large bodies through the Durable Object.** A 22-minute float32 WAV is 466 MB, and the stems
   tar can be over 1 GB in float32 or about 0.6–1.3 GB in FLAC. Workers' request-body limits apply to
   client requests; whether streaming R2 -> container -> R2 through `getTcpPort().fetch()` has
   its own cap is not documented. If it does, have Fretboard upload m4a or FLAC rather than float
   WAV, or let the container fetch from and upload to presigned R2 URLs (needs
   `enableInternet: true`).
4. **Memory in the container since the streamed output: answered.** The 22-minute ensemble's
   separator peaks at 1388 MiB in the container (7131 before; bare-metal Linux 1561 with
   Fretboard's options). Left open: the cgroup peak is mostly reclaimable page cache from the
   rescale temp files (6694 MiB at the default clip mode); a run on a real 3–6 GiB instance
   would show whether Cloudflare's limit treats it that way. If not, drop the temp files' pages
   as they are written (`posix_fadvise(DONTNEED)`), or use `--clip-mode clamp`.
5. **Image storage across deploys.** It is 141 MB per version against 50 GB per account. Whether
   unchanged layers (the weights) are stored once across versions is not documented.
6. **Mel-RoFormer** adds 456 MB of weights (fp16) and its own CPU and memory profile. None of it
   is measured in the container yet. The `melroformer` target is ready for that.
