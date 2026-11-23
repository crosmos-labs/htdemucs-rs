//! `htdemucs-rs separate`: demucs' `separate` command line, in Rust.
//!
//!     htdemucs-rs separate -n htdemucs_6s --shifts 2 --seed 0 --flac --clip-mode clamp -o out song.m4a
//!
//! Writes <out>/<model>/<track name>/<source>.<ext>, like demucs.

use std::path::PathBuf;
use std::time::Instant;

use htdemucs::apply::{self, Options};
use htdemucs::audio::{self, Format};
use htdemucs::{HTDemucs, Tensor};
use rayon::prelude::*;

const USAGE: &str = "usage: htdemucs-rs separate -n <htdemucs|htdemucs_6s|ensemble|melroformer> [options] <tracks>...

options (demucs' names and defaults where they exist):
  -n, --name NAME        model (default htdemucs), or ensemble: htdemucs and htdemucs_6s on one
                         decode, averaged; writes DIR/htdemucs/<track>/ (4 stems) and
                         DIR/htdemucs_6s/<track>/ (6 stems) - see src/ensemble.rs;
                         or melroformer: Kim's Mel-Band RoFormer, vocals + instrumental
  --weights DIR          converted weights (default $HTDEMUCS_WEIGHTS, else ./weights)
  -o, --out DIR          output root (default separated); files go to DIR/NAME/<track>/<stem>.<ext>
  --shifts N             random time shifts averaged (default 1)
  --seed S               seed of the shifts' random offsets, as random.seed(S) in Python (default 0)
  --overlap F            overlap between segments (default 0.25)
  --num-overlap N        melroformer only: chunks overlapping each position (default 2, as upstream)
  --threads T            worker threads (default: all cores)
  --float32              write float32 WAV (default: 16-bit WAV)
  --flac                 write 16-bit FLAC
  --mp3                  not supported: write FLAC/WAV and encode afterwards
  --clip-mode MODE       rescale (default) | clamp | none
  --clip-mode-4 MODE     ensemble only: clip mode of the 4-stem view (default: --clip-mode)
  --ensemble-4 VIEW      ensemble only: the 4-stem view is the average (mean, default) or
                         htdemucs' own stems (htdemucs); the 6-stem set is the ensemble either way
  --ensemble-own-6s      ensemble only: also write htdemucs_6s' own stems (not averaged) to
                         DIR/htdemucs_6s.own/<track>/ (--clip-mode)
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
    clip4: Option<String>,
    average_four: bool,
    own_6s: bool,
    num_overlap: Option<usize>,
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
        clip4: None,
        average_four: true,
        own_6s: false,
        num_overlap: None,
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
            "--num-overlap" => a.num_overlap = Some(num(val(&mut it, &arg)?, &arg)? as usize),
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
            "--clip-mode-4" => {
                let m = val(&mut it, &arg)?;
                if !["rescale", "clamp", "none"].contains(&m.as_str()) {
                    return Err(format!("--clip-mode-4: {m} (rescale, clamp or none)"));
                }
                a.clip4 = Some(m);
            }
            "--ensemble-own-6s" => a.own_6s = true,
            "--ensemble-4" => {
                a.average_four = match val(&mut it, &arg)?.as_str() {
                    "mean" => true,
                    "htdemucs" => false,
                    m => return Err(format!("--ensemble-4: {m} (mean or htdemucs)")),
                };
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
    if a.threads == Some(0) {
        return Err("--threads must be at least 1".into());
    }
    Ok(a)
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
    if a.name == "melroformer" {
        return run_melroformer(a);
    }
    let t0 = Instant::now();
    // `-n ensemble`: htdemucs_6s and htdemucs on one decode, averaged (src/ensemble.rs).
    let ensemble = a.name == "ensemble";
    let names: &[&str] = if ensemble { &["htdemucs_6s", "htdemucs"] } else { &[a.name.as_str()] };
    let models = names.iter().map(|n| HTDemucs::load(&a.weights, n)).collect::<Result<Vec<_>, _>>()?;
    for m in &models {
        apply::stride(m, a.opts.overlap).map_err(|e| format!("--overlap: {e}"))?;
    }
    let rate = models[0].cfg.samplerate as u32;
    let ch = models[0].cfg.audio_channels;
    // Python's `random` as demucs sees it: seeded once, then carried on across the tracks of one
    // run (demucs.separate separates them one after another with the same generator). One per
    // model, so each model's stems are exactly what its own demucs run would give.
    let mut rngs: Vec<_> = models.iter().map(|_| htdemucs::pyrandom::PyRandom::new(a.opts.seed)).collect();
    for track in &a.tracks {
        let t1 = Instant::now();
        let input = audio::decode(track)?;
        let input = audio::resample(audio::to_channels(input, ch)?, rate)?;
        let len = input.frames();
        if a.verbose {
            eprintln!("{}: {:.1} s of audio, decoded in {:.2}s", track.display(), len as f64 / rate as f64, t1.elapsed().as_secs_f64());
        }
        let total: usize = models.iter().zip(&rngs).map(|(m, r)| apply::forwards(m, len, &a.opts, r)).sum();
        let mut done = 0;
        let t2 = Instant::now();
        let show = a.progress;
        let stride_sec = ((1.0 - a.opts.overlap) * models[0].cfg.training_length as f32) as f64 / rate as f64;
        if show {
            tqdm_line(0, total, stride_sec, 0.0);
        }
        let mut progress = || {
            done += 1;
            if show {
                tqdm_line(done, total, stride_sec, t2.elapsed().as_secs_f64());
            }
        };
        // Separated as a stream (apply::Streamer): finished samples go to the writers as soon as
        // every model's overlap-add window has passed them, so nothing track-long is held but
        // the input. The ensemble is averaged window by window.
        let mut streams: Vec<apply::Streamer> = models.iter().zip(rngs.iter_mut()).map(|(m, r)| apply::Streamer::new(m, &input.data, ch, &a.opts, r)).collect();
        if streams[0].silent() {
            eprintln!("warning: {}: the input is silent (or constant): writing silent stems", track.display());
        }
        let name = track.file_stem().and_then(|s| s.to_str()).unwrap_or("track");
        let set = |folder: &str, sources: &[String], clip: &str| -> Result<Vec<audio::StemWriter>, String> {
            let dir = a.out.join(folder).join(name);
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let clip = audio::Clip::parse(clip).ok_or(format!("clip mode {clip}"))?;
            sources.iter().map(|src| audio::StemWriter::create(&dir.join(format!("{src}.{}", a.format.ext())), ch, len, rate, a.format, clip)).collect()
        };
        // Writer sets, in the order windows are written: one set, or the ensemble's 4-stem view
        // and 6-stem set (and htdemucs_6s' own stems).
        let mut sets = if ensemble {
            let mut v = vec![set("htdemucs", models[1].sources(), a.clip4.as_deref().unwrap_or(&a.clip))?, set("htdemucs_6s", models[0].sources(), &a.clip)?];
            if a.own_6s {
                v.push(set("htdemucs_6s.own", models[0].sources(), &a.clip)?);
            }
            v
        } else {
            vec![set(a.name.as_str(), models[0].sources(), &a.clip)?]
        };
        let stride = apply::stride(&models[0], a.opts.overlap)?;
        let mut bufs: Vec<Vec<f32>> = vec![vec![]; streams.len()];
        loop {
            let ready = streams.iter().map(|s| s.ready()).min().unwrap_or(len);
            let taken = streams[0].taken();
            if ready > taken && (ready - taken >= stride || ready == len) {
                let upto = ready.min(taken + stride);
                let w = upto - taken;
                for (s, b) in streams.iter_mut().zip(bufs.iter_mut()) {
                    s.take(upto, b);
                }
                if ensemble {
                    let (n6, n4) = (models[0].sources().len(), models[1].sources().len());
                    if a.own_6s {
                        write_window(&mut sets[2], &bufs[0], ch, w)?;
                    }
                    let six = Tensor::new(&[n6, ch, w], std::mem::take(&mut bufs[0]));
                    let four = Tensor::new(&[n4, ch, w], std::mem::take(&mut bufs[1]));
                    let (v4, v6) = htdemucs::ensemble::combine_with(four, models[1].sources(), six, models[0].sources(), a.average_four)?;
                    write_window(&mut sets[0], &v4.data, ch, w)?;
                    write_window(&mut sets[1], &v6.data, ch, w)?;
                    (bufs[0], bufs[1]) = (v6.data, v4.data);
                } else {
                    write_window(&mut sets[0], &bufs[0], ch, w)?;
                }
                continue;
            }
            if taken == len {
                break;
            }
            // The model furthest behind runs its next segment.
            let k = (0..streams.len()).min_by_key(|&k| streams[k].ready()).unwrap_or(0);
            if !streams[k].step(&mut progress) {
                return Err("internal: a stream stopped short of the track's end".into());
            }
        }
        drop(streams);
        drop(input);
        if show {
            eprintln!();
        }
        if a.verbose {
            eprintln!("  separated in {:.1}s", t2.elapsed().as_secs_f64());
        }
        // Rescaled stems are divided and encoded now (their peak is known only at the end).
        let t3 = Instant::now();
        sets.into_iter().flatten().collect::<Vec<_>>().into_par_iter().try_for_each(|w| w.finish())?;
        if a.verbose {
            eprintln!("  finished writing in {:.1}s, {:.1}s total", t3.elapsed().as_secs_f64(), t1.elapsed().as_secs_f64());
        }
    }
    if a.verbose {
        eprintln!("done in {:.1}s, peak RSS {:.0} MB", t0.elapsed().as_secs_f64(), peak_rss_mb());
    }
    Ok(())
}

/// Hands a window of stems (sources, channels, w) to their writers, stems in parallel.
fn write_window(writers: &mut [audio::StemWriter], data: &[f32], ch: usize, w: usize) -> Result<(), String> {
    writers.par_iter_mut().enumerate().try_for_each(|(i, wr)| {
        let planes: Vec<&[f32]> = (0..ch).map(|c| &data[(i * ch + c) * w..(i * ch + c + 1) * w]).collect();
        wr.push(&planes)
    })
}

/// `-n melroformer`: Kim's Mel-Band RoFormer (src/roformer.rs) with upstream's chunking; writes
/// vocals and instrumental (mix - vocals, as upstream's inference.py) to DIR/melroformer/<track>/.
fn run_melroformer(a: Args) -> Result<(), String> {
    use htdemucs::roformer::{self, Chunking, MelRoformer};
    if a.opts.shifts > 1 || a.opts.overlap != Options::default().overlap {
        return Err("melroformer takes no --shifts / --overlap (it is deterministic; see --num-overlap)".into());
    }
    let t0 = Instant::now();
    let model = MelRoformer::load(&a.weights, "melroformer")?;
    let ch = Chunking::new(&model.cfg, a.num_overlap.unwrap_or(model.cfg.num_overlap)).map_err(|e| format!("--num-overlap: {e}"))?;
    let rate = model.cfg.sample_rate as u32;
    for track in &a.tracks {
        let t1 = Instant::now();
        let input = audio::decode(track)?;
        let input = audio::resample(audio::to_channels(input, 2)?, rate)?;
        let len = input.frames();
        if a.verbose {
            eprintln!("{}: {:.1} s of audio, decoded in {:.2}s", track.display(), len as f64 / rate as f64, t1.elapsed().as_secs_f64());
        }
        let (total, mut done) = (ch.chunks(len), 0);
        let (t2, show, step_sec) = (Instant::now(), a.progress, ch.step as f64 / rate as f64);
        if show {
            tqdm_line(0, total, step_sec, 0.0);
        }
        // Both stems are written as the chunks finalise them (audio::StemWriter, as the Demucs
        // path): vocals, and instrumental = mix - vocals (upstream inference.py).
        let name = track.file_stem().and_then(|s| s.to_str()).unwrap_or("track");
        let dir = a.out.join("melroformer").join(name);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let clip = audio::Clip::parse(&a.clip).ok_or(format!("clip mode {}", a.clip))?;
        let mut writers = ["vocals", "instrumental"]
            .iter()
            .map(|src| audio::StemWriter::create(&dir.join(format!("{src}.{}", a.format.ext())), 2, len, rate, a.format, clip))
            .collect::<Result<Vec<_>, _>>()?;
        let (mut err, mut inst) = (None, vec![]);
        let mix = &input.data;
        let model_fn = |part: &[f32]| model.forward(part, &mut htdemucs::Tracer::none());
        roformer::separate_stream(
            mix,
            &ch,
            &model_fn,
            &mut || {
                done += 1;
                if show {
                    tqdm_line(done, total, step_sec, t2.elapsed().as_secs_f64());
                }
            },
            &mut |t0, l, r| {
                if err.is_some() {
                    return;
                }
                inst.clear();
                inst.extend(l.iter().enumerate().map(|(i, v)| mix[t0 + i] - v));
                inst.extend(r.iter().enumerate().map(|(i, v)| mix[len + t0 + i] - v));
                let (il, ir) = inst.split_at(l.len());
                err = writers[0].push(&[l, r]).and_then(|_| writers[1].push(&[il, ir])).err();
            },
        );
        if let Some(e) = err {
            return Err(e);
        }
        for w in writers {
            w.finish()?;
        }
        if show {
            eprintln!();
        }
        if a.verbose {
            eprintln!("  separated and written in {:.1}s", t2.elapsed().as_secs_f64());
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
