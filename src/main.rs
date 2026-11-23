//! `htdemucs-rs separate`: demucs' `separate` command line, in Rust.
//!
//!     htdemucs-rs separate -n htdemucs_6s --shifts 2 --seed 0 --flac --clip-mode clamp -o out song.m4a
//!
//! Writes <out>/<model>/<track name>/<source>.<ext>, like demucs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use htdemucs::apply::{self, Options};
use htdemucs::audio::{self, Format};
use htdemucs::HTDemucs;

const USAGE: &str = "usage: htdemucs-rs separate -n <htdemucs|htdemucs_6s> [options] <tracks>...

options (demucs' names and defaults where they exist):
  -n, --name NAME        model (default htdemucs)
  --weights DIR          converted weights (default $HTDEMUCS_WEIGHTS, else ./weights)
  -o, --out DIR          output root (default separated); files go to DIR/NAME/<track>/<stem>.<ext>
  --shifts N             random time shifts averaged (default 1)
  --seed S               seed of the shifts' random offsets, as random.seed(S) in Python (default 0)
  --overlap F            overlap between segments (default 0.25)
  --threads T            worker threads (default: all cores)
  --float32              write float32 WAV (default: 16-bit WAV)
  --flac                 write 16-bit FLAC
  --mp3                  not supported: write FLAC/WAV and encode afterwards
  --clip-mode MODE       rescale (default) | clamp | none
  -q, --quiet            no progress bar (by default a tqdm-style bar on stderr, as demucs prints)
  -v, --verbose          timings and peak memory on stderr";

struct Args {
    name: String,
    weights: PathBuf,
    out: PathBuf,
    opts: Options,
    threads: Option<usize>,
    format: Format,
    clip: String,
    verbose: bool,
    progress: bool,
    tracks: Vec<PathBuf>,
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args {
        name: "htdemucs".into(),
        weights: std::env::var_os("HTDEMUCS_WEIGHTS").map(PathBuf::from).unwrap_or_else(|| "weights".into()),
        out: "separated".into(),
        opts: Options::default(),
        threads: None,
        format: Format::Wav16,
        clip: "rescale".into(),
        verbose: false,
        progress: true,
        tracks: vec![],
    };
    let val = |it: &mut dyn Iterator<Item = String>, flag: &str| it.next().ok_or(format!("{flag} needs a value"));
    let num = |s: String, flag: &str| -> Result<f64, String> { s.parse::<f64>().map_err(|_| format!("{flag}: not a number: {s}")) };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-n" | "--name" => a.name = val(&mut it, &arg)?,
            "--weights" => a.weights = val(&mut it, &arg)?.into(),
            "-o" | "--out" => a.out = val(&mut it, &arg)?.into(),
            "--shifts" => a.opts.shifts = num(val(&mut it, &arg)?, &arg)? as usize,
            "--seed" => a.opts.seed = val(&mut it, &arg)?.parse().map_err(|_| "--seed: not an integer".to_string())?,
            "--overlap" => a.opts.overlap = num(val(&mut it, &arg)?, &arg)? as f32,
            "--threads" | "-j" => a.threads = Some(num(val(&mut it, &arg)?, &arg)? as usize),
            "--float32" => a.format = Format::WavF32,
            "--flac" => a.format = Format::Flac16,
            "--mp3" => return Err("--mp3 is not supported: write --flac (or WAV) and encode with ffmpeg / LAME".into()),
            "--clip-mode" => {
                a.clip = val(&mut it, &arg)?;
                if !["rescale", "clamp", "none"].contains(&a.clip.as_str()) {
                    return Err(format!("--clip-mode: {} (rescale, clamp or none)", a.clip));
                }
            }
            "-v" | "--verbose" => a.verbose = true,
            "-q" | "--quiet" => a.progress = false,
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with('-') => return Err(format!("unknown option {s}")),
            _ => a.tracks.push(arg.into()),
        }
    }
    if a.tracks.is_empty() {
        return Err("no tracks given".into());
    }
    if !(0.0..1.0).contains(&a.opts.overlap) {
        return Err("--overlap must be in [0, 1)".into());
    }
    Ok(a)
}

/// demucs.audio.prevent_clip, per stem (all channels): rescale divides by max(1.01 * peak, 1),
/// clamp limits to +-0.99.
fn prevent_clip(x: &mut [f32], mode: &str) {
    match mode {
        "rescale" => {
            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = (1.01f32 * peak).max(1.0);
            x.iter_mut().for_each(|v| *v /= d);
        }
        "clamp" => x.iter_mut().for_each(|v| *v = v.clamp(-0.99, 0.99)),
        _ => {}
    }
}

/// Exactly how many segment forwards `apply::separate` will run: replays the shift offsets with
/// the same CPython-compatible generator (each pass: randint, then one randrange(1) per segment).
fn segment_count(model: &HTDemucs, len: usize, opts: &Options) -> usize {
    let seg = model.cfg.training_length;
    let stride = ((1.0 - opts.overlap) * seg as f32) as usize;
    if opts.shifts == 0 {
        return len.div_ceil(stride);
    }
    let ms = model.cfg.samplerate / 2;
    let mut rng = htdemucs::pyrandom::PyRandom::new(opts.seed);
    let mut total = 0;
    for _ in 0..opts.shifts {
        let offset = rng.randint(0, ms as u32) as usize;
        let n = (len + ms - offset).div_ceil(stride);
        for _ in 0..n {
            rng.randrange(1);
        }
        total += n;
    }
    total
}

fn mmss(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// One progress line in tqdm's format, as demucs prints it (unit: seconds of audio, a segment
/// counting as one stride): ` 42%|████▏     | 122.9/292.5 [00:41<00:57,  2.97seconds/s]`.
/// Fretboard's parser reads the last "done/total [". Unlike demucs, whose bar restarts for each
/// shift pass, this one bar covers every pass, so it only goes forward.
fn tqdm_line(done: usize, total: usize, stride_sec: f64, elapsed: f64) {
    let (d, t) = (done as f64 * stride_sec, total as f64 * stride_sec);
    let frac = if total > 0 { done as f64 / total as f64 } else { 1.0 };
    let filled = (frac * 10.0).floor() as usize;
    let bar: String = "█".repeat(filled) + &" ".repeat(10 - filled.min(10));
    let rate = if elapsed > 0.0 { d / elapsed } else { 0.0 };
    let eta = if done > 0 { elapsed / done as f64 * (total - done) as f64 } else { 0.0 };
    eprint!("\r{:3.0}%|{bar}| {d:.1}/{t:.1} [{}<{}, {rate:6.2}seconds/s]", frac * 100.0, mmss(elapsed), if done > 0 { mmss(eta) } else { "?".into() });
}

fn peak_rss_mb() -> f64 {
    #[repr(C)]
    struct Timeval {
        s: i64,
        us: i64,
    }
    #[repr(C)]
    struct Rusage {
        utime: Timeval,
        stime: Timeval,
        maxrss: i64,
        rest: [i64; 13],
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut r: Rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut r) };
    // ru_maxrss: bytes on macOS, kilobytes on Linux.
    if cfg!(target_os = "macos") { r.maxrss as f64 / 1048576.0 } else { r.maxrss as f64 / 1024.0 }
}

fn run(a: Args) -> Result<(), String> {
    if let Some(t) = a.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().map_err(|e| e.to_string())?;
    }
    let t0 = Instant::now();
    let model = HTDemucs::load(&a.weights, &a.name)?;
    let rate = model.cfg.samplerate as u32;
    let ch = model.cfg.audio_channels;
    for track in &a.tracks {
        let t1 = Instant::now();
        let input = audio::decode(track)?;
        let input = audio::resample(audio::to_channels(input, ch)?, rate)?;
        let len = input.frames();
        if a.verbose {
            eprintln!("{}: {:.1} s of audio, decoded in {:.2}s", track.display(), len as f64 / rate as f64, t1.elapsed().as_secs_f64());
        }
        let total = segment_count(&model, len, &a.opts);
        let mut done = 0;
        let t2 = Instant::now();
        let show = a.progress;
        let stride_sec = ((1.0 - a.opts.overlap) * model.cfg.training_length as f32) as f64 / rate as f64;
        if show {
            tqdm_line(0, total, stride_sec, 0.0);
        }
        let mut progress = || {
            done += 1;
            if show {
                tqdm_line(done, total, stride_sec, t2.elapsed().as_secs_f64());
            }
        };
        let mut stems = apply::separate(&model, &input.data, ch, &a.opts, &mut progress);
        if show {
            eprintln!();
        }
        if a.verbose {
            eprintln!("  separated in {:.1}s", t2.elapsed().as_secs_f64());
        }
        let name = track.file_stem().and_then(|s| s.to_str()).unwrap_or("track");
        let dir = a.out.join(&a.name).join(name);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let plane = ch * len;
        for (i, src) in model.sources().iter().enumerate() {
            let x = &mut stems.data[i * plane..(i + 1) * plane];
            prevent_clip(x, &a.clip);
            let path = dir.join(format!("{src}.{}", a.format.ext()));
            audio::write(Path::new(&path), x, ch, rate, a.format)?;
        }
        if a.verbose {
            eprintln!("  wrote {} in {:.1}s total", dir.display(), t1.elapsed().as_secs_f64());
        }
    }
    if a.verbose {
        eprintln!("done in {:.1}s, peak RSS {:.0} MB", t0.elapsed().as_secs_f64(), peak_rss_mb());
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    if cmd.as_deref() != Some("separate") {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    match parse(args) {
        Ok(a) => {
            if let Err(e) = run(a) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
