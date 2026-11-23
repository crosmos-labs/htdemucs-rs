//! `htdemucs-serve`: the container entrypoint. A thin wrapper that runs the unchanged
//! `htdemucs-rs separate` CLI as a child process, one job at a time, and hands back its stems as
//! a tar archive. Standard library + libc only. See docs/container.md.
//!
//!     htdemucs-serve                       HTTP on $PORT (default 8080)
//!     htdemucs-serve once [key=value]...   one job: audio on stdin, tar of stems on stdout
//!
//! HTTP (one job per container; the Worker / Durable Object polls):
//!
//!     GET    /health          200 "ok" once listening (the readiness check)
//!     POST   /job?key=value   body = the audio file (WAV / FLAC / MP3 / M4A); 202 + status JSON,
//!                             409 while a job is running, 400 on a bad option
//!     GET    /job             status JSON: state idle | receiving | running | done | failed,
//!                             progress, CPU seconds, peak RSS, the host's CPU model
//!     GET    /job/result      the stems as an uncompressed tar (<output folder>/<stem>.<ext>),
//!                             with Content-Length; 409 unless the job is done
//!     DELETE /job             kills a running job, or deletes a finished job's files
//!
//! Options (query string or `once` arguments) map one-to-one onto the CLI's flags: name (or n),
//! shifts, seed, overlap, num-overlap, clip-mode, clip-mode-4, ensemble-4, threads,
//! ensemble-own-6s (=1); plus format = flac (default) | wav (16-bit) | float32.
//!
//! Each job is its own process, so its memory goes back to the system when it ends (the CLI tunes
//! glibc malloc to keep freed memory in the heap), and a failed job cannot take the server down.
//! SIGTERM / SIGINT kill the running job and exit at once: as PID 1 the server would otherwise
//! ignore SIGTERM, and Cloudflare waits up to 15 minutes before its SIGKILL.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Pid of the running separation (0: none), for the signal handler.
static CHILD: AtomicI32 = AtomicI32::new(0);

/// Option names accepted from a request, and the CLI flag each one becomes.
const OPTIONS: &[(&str, &str)] = &[
    ("name", "-n"),
    ("n", "-n"),
    ("shifts", "--shifts"),
    ("seed", "--seed"),
    ("overlap", "--overlap"),
    ("num-overlap", "--num-overlap"),
    ("clip-mode", "--clip-mode"),
    ("clip-mode-4", "--clip-mode-4"),
    ("ensemble-4", "--ensemble-4"),
    ("threads", "--threads"),
];

/// On / off switches (`key=1` or `key=true` adds the flag, `0` / `false` leaves it out).
const SWITCHES: &[(&str, &str)] = &[("ensemble-own-6s", "--ensemble-own-6s")];

/// Request options -> CLI arguments. Values are plain tokens (no leading '-'), so a request can set
/// only the listed flags; the CLI validates the values themselves.
fn cli_args(pairs: &[(String, String)]) -> Result<Vec<String>, String> {
    let mut args = vec![];
    let mut format = "flac";
    for (k, v) in pairs {
        if v.is_empty() || v.starts_with('-') || !v.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
            return Err(format!("{k}: bad value {v:?}"));
        }
        if k == "format" {
            format = match v.as_str() {
                "flac" => "flac",
                "wav" => "wav",
                "float32" => "float32",
                _ => return Err(format!("format: {v} (flac, wav or float32)")),
            };
            continue;
        }
        if let Some((_, flag)) = SWITCHES.iter().find(|(n, _)| n == k) {
            match v.as_str() {
                "1" | "true" => args.push(flag.to_string()),
                "0" | "false" => {}
                _ => return Err(format!("{k}: {v} (1 / true or 0 / false)")),
            }
            continue;
        }
        let flag = OPTIONS.iter().find(|(n, _)| n == k).map(|(_, f)| *f);
        let Some(flag) = flag else {
            let names: Vec<&str> = OPTIONS.iter().chain(SWITCHES).map(|(n, _)| *n).chain(["format"]).collect();
            return Err(format!("unknown option {k} (allowed: {})", names.join(", ")));
        };
        args.extend([flag.to_string(), v.clone()]);
    }
    match format {
        "flac" => args.push("--flac".into()),
        "float32" => args.push("--float32".into()),
        _ => {}
    }
    Ok(args)
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Receiving,
    Running,
    Done,
    Failed,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            State::Receiving => "receiving",
            State::Running => "running",
            State::Done => "done",
            State::Failed => "failed",
        }
    }
}

struct Job {
    id: u64,
    args: Vec<String>,
    state: State,
    error: String,
    /// Progress in seconds of audio processed / total (the CLI's tqdm line; every shift pass and
    /// both ensemble models count).
    done: f64,
    total: f64,
    started: Instant,
    wall: f64,
    cpu: f64,
    peak_rss_mb: f64,
    input_bytes: u64,
    dir: PathBuf,
    files: Vec<(String, PathBuf, u64)>,
}

impl Job {
    fn new(id: u64, args: Vec<String>, dir: PathBuf) -> Job {
        Job {
            id,
            args,
            state: State::Receiving,
            error: String::new(),
            done: 0.0,
            total: 0.0,
            started: Instant::now(),
            wall: 0.0,
            cpu: 0.0,
            peak_rss_mb: 0.0,
            input_bytes: 0,
            dir,
            files: vec![],
        }
    }
}

struct Shared {
    job: Option<Job>,
    next_id: u64,
}

struct Env {
    bin: PathBuf,
    root: PathBuf,
    max_upload: u64,
    cpu_model: String,
    nproc: usize,
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn status_json(job: Option<&Job>, env: &Env) -> String {
    let host = format!("\"cpu_model\":{},\"nproc\":{}", esc(&env.cpu_model), env.nproc);
    let Some(j) = job else {
        return format!("{{\"state\":\"idle\",{host}}}\n");
    };
    let wall = if matches!(j.state, State::Done | State::Failed) { j.wall } else { j.started.elapsed().as_secs_f64() };
    let args: Vec<String> = j.args.iter().map(|a| esc(a)).collect();
    let files: Vec<String> = j.files.iter().map(|(n, _, b)| format!("{{\"name\":{},\"bytes\":{b}}}", esc(n))).collect();
    format!(
        "{{\"state\":\"{}\",\"id\":{},\"args\":[{}],\"input_bytes\":{},\"progress\":{{\"done\":{:.1},\"total\":{:.1}}},\"wall_s\":{:.2},\"cpu_s\":{:.2},\"peak_rss_mb\":{:.0},\"error\":{},\"files\":[{}],{host}}}\n",
        j.state.name(),
        j.id,
        args.join(","),
        j.input_bytes,
        j.done,
        j.total,
        wall,
        j.cpu,
        j.peak_rss_mb,
        esc(&j.error),
        files.join(","),
    )
}

/// The input's extension from its first bytes (a hint for the decoder, which also probes).
fn sniff(head: &[u8]) -> &'static str {
    if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WAVE" {
        "wav"
    } else if head.starts_with(b"fLaC") {
        "flac"
    } else if head.len() >= 8 && &head[4..8] == b"ftyp" {
        "m4a"
    } else if head.starts_with(b"ID3") || (head.len() >= 2 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0) {
        "mp3"
    } else {
        "audio"
    }
}

/// Moves `dir/in/upload` to `dir/in/track.<ext>` (the CLI names the output folder after the
/// file stem, so every job's stems land in `<model>/track/`).
fn name_input(dir: &Path) -> io::Result<PathBuf> {
    let up = dir.join("in/upload");
    let mut head = [0u8; 16];
    let n = File::open(&up)?.read(&mut head)?;
    let path = dir.join(format!("in/track.{}", sniff(&head[..n])));
    fs::rename(&up, &path)?;
    Ok(path)
}

fn spawn(env: &Env, args: &[String], dir: &Path, input: &Path) -> io::Result<Child> {
    let child = Command::new(&env.bin)
        .arg("separate")
        .args(args)
        .arg("-v")
        .arg("-o")
        .arg(dir.join("out"))
        .arg(input)
        .stdin(Stdio::null())
        .stdout(Stdio::from(io::stderr()))
        .stderr(Stdio::piped())
        .spawn()?;
    CHILD.store(child.id() as i32, Ordering::SeqCst);
    Ok(child)
}

struct Outcome {
    ok: bool,
    error: String,
    cpu: f64,
    peak_rss_mb: f64,
}

/// Follows the child's stderr (progress bar -> `progress`, other lines -> our log), then reaps it
/// with wait4 for its exit status, CPU time and peak RSS.
fn supervise(mut child: Child, tag: &str, mut progress: impl FnMut(f64, f64)) -> Outcome {
    let pid = child.id() as i32;
    let mut err = child.stderr.take().unwrap();
    let mut tail: Vec<String> = vec![];
    let (mut buf, mut chunk) = (Vec::new(), [0u8; 4096]);
    let mut logged_pct = -1i64;
    let mut line = |seg: &[u8], tail: &mut Vec<String>| {
        let s = String::from_utf8_lossy(seg);
        let s = s.trim();
        if s.is_empty() {
            return;
        }
        // tqdm-style: ` 42%|████▏     | 122.9/292.5 [00:41<00:57,  2.97seconds/s]`
        if s.ends_with("seconds/s]") {
            if let Some((d, t)) = s.rsplit_once(" [").and_then(|(a, _)| a.rsplit_once("| ")).and_then(|(_, f)| f.split_once('/')) {
                if let (Ok(d), Ok(t)) = (d.trim().parse::<f64>(), t.trim().parse::<f64>()) {
                    progress(d, t);
                    let pct = if t > 0.0 { (d / t * 10.0).floor() as i64 * 10 } else { 0 };
                    if pct != logged_pct {
                        logged_pct = pct;
                        eprintln!("[{tag}] {pct}% ({d:.0}/{t:.0} s of audio)");
                    }
                }
            }
            return;
        }
        eprintln!("[{tag}] {s}");
        tail.push(s.to_string());
        if tail.len() > 20 {
            tail.remove(0);
        }
    };
    loop {
        match err.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                while let Some(p) = buf.iter().position(|&b| b == b'\r' || b == b'\n') {
                    let seg: Vec<u8> = buf.drain(..=p).collect();
                    line(&seg[..seg.len() - 1], &mut tail);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    line(&buf, &mut tail);
    let mut status = 0;
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    loop {
        let r = unsafe { libc::wait4(pid, &mut status, 0, &mut ru) };
        if r == pid || (r < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted) {
            break;
        }
    }
    CHILD.store(0, Ordering::SeqCst);
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    let cpu = tv(ru.ru_utime) + tv(ru.ru_stime);
    // ru_maxrss: kilobytes on Linux, bytes on macOS.
    let peak_rss_mb = if cfg!(target_os = "macos") { ru.ru_maxrss as f64 / 1048576.0 } else { ru.ru_maxrss as f64 / 1024.0 };
    let (sig, code) = (status & 0x7f, (status >> 8) & 0xff);
    let ok = sig == 0 && code == 0;
    let error = if ok {
        String::new()
    } else {
        let how = if sig != 0 { format!("killed by signal {sig}") } else { format!("exit code {code}") };
        format!("htdemucs-rs {how}: {}", tail.join("\n"))
    };
    Outcome { ok, error, cpu, peak_rss_mb }
}

/// The stems the CLI wrote: `out/<model folder>/track/<stem>.<ext>`, as (tar name, path, bytes).
fn stems(dir: &Path) -> io::Result<Vec<(String, PathBuf, u64)>> {
    let mut v = vec![];
    for folder in fs::read_dir(dir.join("out"))? {
        let folder = folder?;
        let model = folder.file_name().to_string_lossy().into_owned();
        for track in fs::read_dir(folder.path())? {
            for f in fs::read_dir(track?.path())? {
                let f = f?;
                v.push((format!("{model}/{}", f.file_name().to_string_lossy()), f.path(), f.metadata()?.len()));
            }
        }
    }
    v.sort();
    Ok(v)
}

fn tar_len(files: &[(String, PathBuf, u64)]) -> u64 {
    files.iter().map(|f| 512 + f.2.div_ceil(512) * 512).sum::<u64>() + 1024
}

fn tar_header(name: &str, size: u64) -> [u8; 512] {
    let mut h = [0u8; 512];
    let put = |h: &mut [u8; 512], at: usize, s: &str| h[at..at + s.len()].copy_from_slice(s.as_bytes());
    let name = &name.as_bytes()[..name.len().min(99)];
    h[..name.len()].copy_from_slice(name);
    let mtime = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    put(&mut h, 100, "0000644\0");
    put(&mut h, 108, "0000000\0");
    put(&mut h, 116, "0000000\0");
    put(&mut h, 124, &format!("{size:011o}\0"));
    put(&mut h, 136, &format!("{mtime:011o}\0"));
    put(&mut h, 148, "        ");
    h[156] = b'0';
    put(&mut h, 257, "ustar\0");
    put(&mut h, 263, "00");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    put(&mut h, 148, &format!("{sum:06o}\0 "));
    h
}

fn write_tar(w: &mut impl Write, files: &[(String, PathBuf, u64)]) -> io::Result<()> {
    for (name, path, size) in files {
        w.write_all(&tar_header(name, *size))?;
        let n = io::copy(&mut File::open(path)?.take(*size), w)?;
        if n != *size {
            return Err(io::Error::other(format!("{name}: changed while sending")));
        }
        w.write_all(&vec![0u8; (size.div_ceil(512) * 512 - size) as usize])?;
    }
    w.write_all(&[0u8; 1024])?;
    w.flush()
}

fn new_job_dir(root: &Path, id: u64) -> io::Result<PathBuf> {
    let dir = root.join(format!("job-{id}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("in"))?;
    Ok(dir)
}

// ---------------------------------------------------------------------------------------------
// HTTP

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    length: Option<u64>,
    chunked: bool,
    expect_continue: bool,
}

fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => o.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        o.push(v);
                        i += 2;
                    }
                    Err(_) => o.push(b'%'),
                }
            }
            c => o.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&o).into_owned()
}

fn read_head(r: &mut impl BufRead) -> io::Result<Option<Request>> {
    let mut r = r.take(64 * 1024);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("/").to_string());
    let (path, qs) = target.split_once('?').unwrap_or((&target, ""));
    let query = qs.split('&').filter(|p| !p.is_empty()).map(|p| {
        let (k, v) = p.split_once('=').unwrap_or((p, ""));
        (pct_decode(k), pct_decode(v))
    });
    let mut req = Request { method, path: path.to_string(), query: query.collect(), length: None, chunked: false, expect_continue: false };
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(io::Error::other("headers cut short"));
        }
        let l = line.trim_end();
        if l.is_empty() {
            return Ok(Some(req));
        }
        if let Some((k, v)) = l.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_ascii_lowercase());
            match k.as_str() {
                "content-length" => req.length = Some(v.parse().map_err(|_| io::Error::other("bad content-length"))?),
                "transfer-encoding" => req.chunked = v.contains("chunked"),
                "expect" => req.expect_continue = v == "100-continue",
                _ => {}
            }
        }
    }
}

/// Copies the request body (Content-Length or chunked) to `out`, refusing more than `max` bytes.
fn read_body(r: &mut impl BufRead, req: &Request, out: &mut impl Write, max: u64) -> io::Result<u64> {
    let too_big = || io::Error::other(format!("body larger than {} MB (HTDEMUCS_MAX_UPLOAD_MB)", max >> 20));
    if req.chunked {
        let mut total = 0u64;
        loop {
            let mut size = String::new();
            r.read_line(&mut size)?;
            let hex = size.trim().split(';').next().unwrap_or("");
            let n = u64::from_str_radix(hex, 16).map_err(|_| io::Error::other("bad chunk size"))?;
            if n == 0 {
                loop {
                    // trailers, up to the blank line
                    size.clear();
                    if r.read_line(&mut size)? == 0 || size.trim().is_empty() {
                        return Ok(total);
                    }
                }
            }
            total += n;
            if total > max {
                return Err(too_big());
            }
            if io::copy(&mut r.take(n), out)? != n {
                return Err(io::Error::other("body cut short"));
            }
            let mut crlf = [0u8; 2];
            r.read_exact(&mut crlf)?;
        }
    }
    let n = req.length.ok_or_else(|| io::Error::other("needs Content-Length or chunked encoding"))?;
    if n > max {
        return Err(too_big());
    }
    if io::copy(&mut r.take(n), out)? != n {
        return Err(io::Error::other("body cut short"));
    }
    Ok(n)
}

fn respond(w: &mut impl Write, code: u16, ctype: &str, body: &[u8]) -> io::Result<()> {
    let reason = match code {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    };
    write!(w, "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    w.write_all(body)?;
    w.flush()
}

fn json_error(w: &mut impl Write, code: u16, msg: &str) -> io::Result<()> {
    respond(w, code, "application/json", format!("{{\"error\":{}}}\n", esc(msg)).as_bytes())
}

fn handle(stream: TcpStream, sh: &Arc<Mutex<Shared>>, env: &Env) -> io::Result<()> {
    let mut r = BufReader::with_capacity(1 << 16, stream.try_clone()?);
    let mut w = io::BufWriter::with_capacity(1 << 16, stream);
    let Some(req) = read_head(&mut r)? else { return Ok(()) };
    let status = |sh: &Mutex<Shared>| status_json(sh.lock().unwrap().job.as_ref(), env);
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => respond(&mut w, 200, "text/plain", b"ok\n"),
        ("GET", "/job") => respond(&mut w, 200, "application/json", status(sh).as_bytes()),
        ("POST", "/job") => {
            let args = match cli_args(&req.query) {
                Ok(a) => a,
                Err(e) => return json_error(&mut w, 400, &e),
            };
            if req.length.is_none() && !req.chunked {
                return json_error(&mut w, 411, "send the audio as the body, with Content-Length or chunked encoding");
            }
            let dir = {
                let mut s = sh.lock().unwrap();
                if let Some(j) = &s.job {
                    if matches!(j.state, State::Receiving | State::Running) {
                        drop(s);
                        return respond(&mut w, 409, "application/json", status(sh).as_bytes());
                    }
                    let _ = fs::remove_dir_all(&j.dir);
                }
                s.next_id += 1;
                let id = s.next_id;
                let dir = new_job_dir(&env.root, id)?;
                s.job = Some(Job::new(id, args.clone(), dir.clone()));
                dir
            };
            if req.expect_continue {
                w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
                w.flush()?;
            }
            let received = File::create(dir.join("in/upload")).and_then(|f| {
                let mut f = io::BufWriter::with_capacity(1 << 20, f);
                let n = read_body(&mut r, &req, &mut f, env.max_upload)?;
                f.flush()?;
                Ok(n)
            });
            let started = received.and_then(|n| {
                let input = name_input(&dir)?;
                Ok((n, spawn(env, &args, &dir, &input)?))
            });
            let mut s = sh.lock().unwrap();
            let job = s.job.as_mut().unwrap();
            match started {
                Ok((n, child)) => {
                    job.input_bytes = n;
                    job.state = State::Running;
                    let id = job.id;
                    eprintln!("[job {id}] {n} bytes, htdemucs-rs separate {}", args.join(" "));
                    drop(s);
                    let sh2 = Arc::clone(sh);
                    std::thread::spawn(move || run_job(child, id, sh2, dir));
                    respond(&mut w, 202, "application/json", status(sh).as_bytes())
                }
                Err(e) => {
                    job.state = State::Failed;
                    job.error = e.to_string();
                    job.wall = job.started.elapsed().as_secs_f64();
                    let code = if e.to_string().starts_with("body larger") { 413 } else { 400 };
                    drop(s);
                    eprintln!("[job] refused: {e}");
                    json_error(&mut w, code, &e.to_string())
                }
            }
        }
        ("GET", "/job/result") => {
            let files = {
                let s = sh.lock().unwrap();
                match &s.job {
                    Some(j) if j.state == State::Done => j.files.clone(),
                    _ => {
                        drop(s);
                        return respond(&mut w, 409, "application/json", status(sh).as_bytes());
                    }
                }
            };
            write!(w, "HTTP/1.1 200 OK\r\nContent-Type: application/x-tar\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", tar_len(&files))?;
            write_tar(&mut w, &files)
        }
        ("DELETE", "/job") => {
            let mut s = sh.lock().unwrap();
            match s.job.as_ref().map(|j| j.state) {
                Some(State::Running) => {
                    let pid = CHILD.load(Ordering::SeqCst);
                    if pid > 0 {
                        unsafe { libc::kill(pid, libc::SIGKILL) };
                    }
                }
                Some(State::Receiving) => {}
                Some(_) => {
                    let _ = fs::remove_dir_all(&s.job.as_ref().unwrap().dir);
                    s.job = None;
                }
                None => {}
            }
            drop(s);
            respond(&mut w, 200, "application/json", status(sh).as_bytes())
        }
        (_, "/health" | "/job" | "/job/result") => json_error(&mut w, 405, "method not allowed"),
        _ => json_error(&mut w, 404, "not found"),
    }
}

fn run_job(child: Child, id: u64, sh: Arc<Mutex<Shared>>, dir: PathBuf) {
    let tag = format!("job {id}");
    let out = supervise(child, &tag, |d, t| {
        if let Some(j) = sh.lock().unwrap().job.as_mut() {
            (j.done, j.total) = (d, t);
        }
    });
    let files = if out.ok { stems(&dir) } else { Ok(vec![]) };
    let mut s = sh.lock().unwrap();
    let Some(j) = s.job.as_mut().filter(|j| j.id == id) else { return };
    j.wall = j.started.elapsed().as_secs_f64();
    (j.cpu, j.peak_rss_mb) = (out.cpu, out.peak_rss_mb);
    match files {
        Ok(f) if out.ok => {
            j.files = f;
            j.state = State::Done;
        }
        Ok(_) => {
            j.state = State::Failed;
            j.error = out.error;
        }
        Err(e) => {
            j.state = State::Failed;
            j.error = format!("reading the stems: {e}");
        }
    }
    let _ = fs::remove_dir_all(dir.join("in"));
    eprintln!("[{tag}] {} in {:.1}s, {:.1} CPU s, peak RSS {:.0} MB {}", j.state.name(), j.wall, j.cpu, j.peak_rss_mb, j.error);
}

// ---------------------------------------------------------------------------------------------

extern "C" fn on_signal(sig: libc::c_int) {
    let pid = CHILD.load(Ordering::SeqCst);
    if pid > 0 {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    unsafe { libc::_exit(128 + sig) };
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    unsafe {
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
    }
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let bin = var("HTDEMUCS_BIN").map(PathBuf::from).unwrap_or_else(|| {
        let sib = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("htdemucs-rs")));
        sib.filter(|p| p.exists()).unwrap_or_else(|| "htdemucs-rs".into())
    });
    let root = var("HTDEMUCS_JOBS").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("htdemucs-jobs"));
    let max_upload = var("HTDEMUCS_MAX_UPLOAD_MB").and_then(|v| v.parse::<u64>().ok()).unwrap_or(2048) << 20;
    let env = Env { bin, root, max_upload, cpu_model: cpu_model(), nproc: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        None | Some("serve") => serve(env),
        Some("once") => once(env, &args[1..]),
        _ => {
            eprintln!("usage: htdemucs-serve [serve]                 HTTP on $PORT (default 8080)\n       htdemucs-serve once [key=value]...      audio on stdin -> tar of stems on stdout");
            2
        }
    };
    std::process::exit(code);
}

fn serve(env: Env) -> i32 {
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let _ = fs::remove_dir_all(&env.root);
    let listener = match TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("htdemucs-serve: port {port}: {e}");
            return 1;
        }
    };
    eprintln!("htdemucs-serve: listening on :{port} ({}, {} CPUs, separator {})", env.cpu_model, env.nproc, env.bin.display());
    let sh = Arc::new(Mutex::new(Shared { job: None, next_id: 0 }));
    let env = Arc::new(env);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let (sh, env) = (Arc::clone(&sh), Arc::clone(&env));
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &sh, &env) {
                eprintln!("htdemucs-serve: connection: {e}");
            }
        });
    }
    0
}

fn once(env: Env, kv: &[String]) -> i32 {
    let pairs: Vec<(String, String)> = kv.iter().map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())).unwrap_or((a.clone(), String::new()))).collect();
    let run = || -> Result<Outcome, String> {
        let args = cli_args(&pairs)?;
        let dir = new_job_dir(&env.root, std::process::id() as u64).map_err(|e| e.to_string())?;
        let mut f = File::create(dir.join("in/upload")).map_err(|e| e.to_string())?;
        let n = io::copy(&mut io::stdin().lock(), &mut f).map_err(|e| format!("reading stdin: {e}"))?;
        drop(f);
        let input = name_input(&dir).map_err(|e| e.to_string())?;
        eprintln!("[once] {n} bytes, htdemucs-rs separate {}", args.join(" "));
        let t0 = Instant::now();
        let child = spawn(&env, &args, &dir, &input).map_err(|e| format!("{}: {e}", env.bin.display()))?;
        let out = supervise(child, "once", |_, _| {});
        let wall = t0.elapsed().as_secs_f64();
        let result = if out.ok {
            let files = stems(&dir).map_err(|e| e.to_string())?;
            write_tar(&mut io::BufWriter::with_capacity(1 << 20, io::stdout().lock()), &files).map_err(|e| format!("writing stdout: {e}"))?;
            Ok(())
        } else {
            Err(out.error.clone())
        };
        let _ = fs::remove_dir_all(&dir);
        eprintln!(
            "{{\"state\":\"{}\",\"args\":[{}],\"input_bytes\":{n},\"wall_s\":{wall:.2},\"cpu_s\":{:.2},\"peak_rss_mb\":{:.0},\"cpu_model\":{},\"nproc\":{}}}",
            if out.ok { "done" } else { "failed" },
            args.iter().map(|a| esc(a)).collect::<Vec<_>>().join(","),
            out.cpu,
            out.peak_rss_mb,
            esc(&env.cpu_model),
            env.nproc
        );
        result.map(|_| out)
    };
    match run() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("htdemucs-serve once: {e}");
            1
        }
    }
}
