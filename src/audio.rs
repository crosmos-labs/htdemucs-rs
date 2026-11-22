//! Audio input / output.
//!
//! Input: anything symphonia decodes (WAV PCM/float, FLAC, MP3, AAC in MP4/M4A), converted the
//! way demucs' `load_track` does: to 44.1 kHz (resampled only when needed) and to 2 channels
//! (mono duplicated, more than two channels: the first two).
//!
//! MP4/M4A: symphonia's MP4 reader ignores edit lists, so the encoder delay (priming samples) an
//! AAC file declares in its `elst` box would stay at the start and every time would be shifted.
//! We parse the edit list ourselves and trim exactly as ffmpeg does (skip `media_time` samples,
//! keep `segment_duration`), so decoded audio lines up sample for sample with ffmpeg's.
//!
//! Output: WAV float32 / 16-bit and FLAC 16-bit. The 16-bit conversions reproduce what demucs
//! gets from torchaudio's soundfile backend (libsndfile), measured: WAV takes the top 16 bits of
//! round(x * 2^31) (i.e. floor(x * 32768) after rounding at 32 bits), FLAC rounds x * 32768 to
//! nearest-even; both saturate.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Planar audio: `data` is (channels, frames) row-major.
pub struct Audio {
    pub rate: u32,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.data.len() / self.channels.max(1)
    }
}

pub fn decode(path: &Path) -> Result<Audio, String> {
    let err = |e: SymError| format!("{}: {e}", path.display());
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(err)?;
    let track = format.default_track(TrackType::Audio).ok_or(format!("{}: no audio track", path.display()))?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or("no audio codec parameters")?.clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(err)?;
    let mut rate = params.sample_rate.unwrap_or(0);
    let mut channels = params.channels.as_ref().map(|c| c.count()).unwrap_or(0);
    let mut inter: Vec<f32> = vec![];
    let mut buf: Vec<f32> = vec![];
    while let Some(packet) = format.next_packet().map_err(err)? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(b) => {
                rate = b.spec().rate();
                channels = b.spec().channels().count();
                buf.resize(b.samples_interleaved(), 0.0);
                b.copy_to_slice_interleaved(&mut buf[..]);
                inter.extend_from_slice(&buf);
            }
            // Like ffmpeg, skip a corrupt packet rather than fail the whole file.
            Err(SymError::DecodeError(e)) => eprintln!("warning: {}: skipped a packet: {e}", path.display()),
            Err(e) => return Err(err(e)),
        }
    }
    if channels == 0 || rate == 0 {
        return Err(format!("{}: no audio decoded", path.display()));
    }
    let mut frames = inter.len() / channels;
    let mut start = 0;
    if let Some((skip, keep)) = mp4_edit(path, rate)? {
        start = skip.min(frames);
        frames = keep.map_or(frames - start, |k| k.min(frames - start));
    }
    let mut data = vec![0.0f32; channels * frames];
    for c in 0..channels {
        for i in 0..frames {
            data[c * frames + i] = inter[(start + i) * channels + c];
        }
    }
    Ok(Audio { rate, channels, data })
}

/// demucs' channel conversion: same count: unchanged; 1 -> 2 duplicated; more: the first two.
pub fn to_channels(a: Audio, n: usize) -> Result<Audio, String> {
    let f = a.frames();
    if a.channels == n {
        Ok(a)
    } else if a.channels == 1 {
        let mut data = Vec::with_capacity(n * f);
        for _ in 0..n {
            data.extend_from_slice(&a.data);
        }
        Ok(Audio { rate: a.rate, channels: n, data })
    } else if a.channels > n {
        Ok(Audio { rate: a.rate, channels: n, data: a.data[..n * f].to_vec() })
    } else {
        Err(format!("{} channels: fewer than {n} and not mono", a.channels))
    }
}

/// Band-limited resampling (rubato's synchronous FFT resampler), only used when the input is not
/// already at `rate`. (demucs resamples with ffmpeg's swresample; resampled input can therefore
/// never be bit-identical to demucs', only close.)
pub fn resample(a: Audio, rate: u32) -> Result<Audio, String> {
    use rubato::audioadapter::Adapter;
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{Fft, FixedSync, Resampler};
    if a.rate == rate {
        return Ok(a);
    }
    let frames = a.frames();
    let planes: Vec<Vec<f32>> = a.data.chunks(frames).map(|c| c.to_vec()).collect();
    let input = SequentialSliceOfVecs::new(&planes, a.channels, frames).map_err(|e| e.to_string())?;
    let mut rs = Fft::<f32>::new(a.rate as usize, rate as usize, 1024, a.channels, FixedSync::Input)
        .map_err(|e| e.to_string())?;
    let out = rs.process_all(&input, frames, None).map_err(|e| e.to_string())?;
    let n = out.frames();
    let mut data = vec![0.0f32; a.channels * n];
    for c in 0..a.channels {
        for i in 0..n {
            data[c * n + i] = out.read_sample(c, i).unwrap_or(0.0);
        }
    }
    Ok(Audio { rate, channels: a.channels, data })
}

/// For an MP4/M4A file: (samples to skip, samples to keep) from the audio track's edit list, as
/// ffmpeg applies it. None if not an MP4 or no edit list.
fn mp4_edit(path: &Path, rate: u32) -> Result<Option<(usize, Option<usize>)>, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let len = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut head = [0u8; 8];
    if f.read_exact(&mut head).is_err() || &head[4..8] != b"ftyp" {
        return Ok(None);
    }
    // Find and read the moov box.
    let mut pos = 0u64;
    let mut moov = None;
    while pos + 8 <= len {
        f.seek(SeekFrom::Start(pos)).map_err(|e| e.to_string())?;
        let mut h = [0u8; 16];
        f.read_exact(&mut h[..8]).map_err(|e| e.to_string())?;
        let mut size = u32::from_be_bytes(h[..4].try_into().unwrap()) as u64;
        let mut hdr = 8;
        if size == 1 {
            f.read_exact(&mut h[8..16]).map_err(|e| e.to_string())?;
            size = u64::from_be_bytes(h[8..16].try_into().unwrap());
            hdr = 16;
        } else if size == 0 {
            size = len - pos;
        }
        if &h[4..8] == b"moov" {
            let mut b = vec![0u8; (size - hdr) as usize];
            f.read_exact(&mut b).map_err(|e| e.to_string())?;
            moov = Some(b);
            break;
        }
        if size < 8 {
            break;
        }
        pos += size;
    }
    let Some(moov) = moov else { return Ok(None) };
    let movie_ts = child(&moov, b"mvhd").map(|b| timescale(b)).unwrap_or(0);
    let mut off = 0;
    while let Some((typ, body, next)) = next_box(&moov, off) {
        off = next;
        if typ != *b"trak" {
            continue;
        }
        let Some(mdia) = child(body, b"mdia") else { continue };
        let is_sound = child(mdia, b"hdlr").is_some_and(|h| h.len() >= 12 && &h[8..12] == b"soun");
        if !is_sound {
            continue;
        }
        let media_ts = child(mdia, b"mdhd").map(|b| timescale(b)).unwrap_or(0);
        let Some(elst) = child(body, b"edts").and_then(|e| child(e, b"elst")) else { return Ok(None) };
        let version = elst[0];
        let count = u32::from_be_bytes(elst[4..8].try_into().unwrap());
        let mut p = 8;
        let mut skip = 0i64;
        let mut keep_movie = 0u64;
        for i in 0..count {
            let (dur, mt) = if version == 1 {
                let d = u64::from_be_bytes(elst[p..p + 8].try_into().unwrap());
                let m = i64::from_be_bytes(elst[p + 8..p + 16].try_into().unwrap());
                p += 20;
                (d, m)
            } else {
                let d = u32::from_be_bytes(elst[p..p + 4].try_into().unwrap()) as u64;
                let m = i32::from_be_bytes(elst[p + 4..p + 8].try_into().unwrap()) as i64;
                p += 12;
                (d, m)
            };
            if mt == -1 {
                return Err(format!("{}: MP4 edit lists with empty edits are not supported", path.display()));
            }
            if i == 0 {
                skip = mt;
            } else {
                return Err(format!("{}: MP4 edit lists with several edits are not supported", path.display()));
            }
            keep_movie += dur;
        }
        if media_ts == 0 || movie_ts == 0 {
            return Ok(None);
        }
        let to_samples = |v: u128, ts: u64| ((v * rate as u128 + ts as u128 / 2) / ts as u128) as usize;
        let skip = to_samples(skip.max(0) as u128, media_ts);
        let keep = (keep_movie > 0).then(|| to_samples(keep_movie as u128, movie_ts));
        return Ok(Some((skip, keep)));
    }
    Ok(None)
}

fn timescale(full_box: &[u8]) -> u64 {
    // mvhd / mdhd: version(1) flags(3), then v0: ctime(4) mtime(4) timescale(4); v1: 8, 8, 4.
    if full_box[0] == 1 {
        u32::from_be_bytes(full_box[20..24].try_into().unwrap()) as u64
    } else {
        u32::from_be_bytes(full_box[12..16].try_into().unwrap()) as u64
    }
}

fn next_box(b: &[u8], off: usize) -> Option<([u8; 4], &[u8], usize)> {
    if off + 8 > b.len() {
        return None;
    }
    let mut size = u32::from_be_bytes(b[off..off + 4].try_into().unwrap()) as usize;
    let typ: [u8; 4] = b[off + 4..off + 8].try_into().unwrap();
    let mut hdr = 8;
    if size == 1 {
        size = u64::from_be_bytes(b[off + 8..off + 16].try_into().unwrap()) as usize;
        hdr = 16;
    } else if size == 0 {
        size = b.len() - off;
    }
    if size < hdr || off + size > b.len() {
        return None;
    }
    Some((typ, &b[off + hdr..off + size], off + size))
}

fn child<'a>(b: &'a [u8], typ: &[u8; 4]) -> Option<&'a [u8]> {
    let mut off = 0;
    while let Some((t, body, next)) = next_box(b, off) {
        if &t == typ {
            return Some(body);
        }
        off = next;
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    WavF32,
    Wav16,
    Flac16,
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::WavF32 | Format::Wav16 => "wav",
            Format::Flac16 => "flac",
        }
    }
}

/// libsndfile's float -> 16-bit WAV: round to 32 bits, keep the top 16 (arithmetic shift).
pub fn wav16(x: f32) -> i16 {
    let v = (x as f64 * 2147483648.0).round_ties_even().clamp(-2147483648.0, 2147483647.0) as i32;
    (v >> 16) as i16
}

/// libsndfile's float -> 16-bit FLAC: round(x * 32768) to nearest even, saturated.
pub fn flac16(x: f32) -> i16 {
    (x as f64 * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i16
}

/// Writes planar (channels, frames) audio.
pub fn write(path: &Path, data: &[f32], channels: usize, rate: u32, fmt: Format) -> Result<(), String> {
    let frames = data.len() / channels;
    let e = |e: std::io::Error| format!("{}: {e}", path.display());
    match fmt {
        Format::WavF32 | Format::Wav16 => {
            let float = fmt == Format::WavF32;
            let bytes_ps = if float { 4 } else { 2 };
            let data_len = (frames * channels * bytes_ps) as u32;
            let mut w = BufWriter::new(File::create(path).map_err(e)?);
            let mut h = Vec::with_capacity(44);
            h.extend_from_slice(b"RIFF");
            h.extend_from_slice(&(36 + data_len).to_le_bytes());
            h.extend_from_slice(b"WAVEfmt ");
            h.extend_from_slice(&16u32.to_le_bytes());
            h.extend_from_slice(&(if float { 3u16 } else { 1u16 }).to_le_bytes());
            h.extend_from_slice(&(channels as u16).to_le_bytes());
            h.extend_from_slice(&rate.to_le_bytes());
            h.extend_from_slice(&(rate * (channels * bytes_ps) as u32).to_le_bytes());
            h.extend_from_slice(&((channels * bytes_ps) as u16).to_le_bytes());
            h.extend_from_slice(&((bytes_ps * 8) as u16).to_le_bytes());
            h.extend_from_slice(b"data");
            h.extend_from_slice(&data_len.to_le_bytes());
            w.write_all(&h).map_err(e)?;
            let mut buf = Vec::with_capacity(channels * bytes_ps * 4096);
            for i0 in (0..frames).step_by(4096) {
                buf.clear();
                for i in i0..(i0 + 4096).min(frames) {
                    for c in 0..channels {
                        let x = data[c * frames + i];
                        if float {
                            buf.extend_from_slice(&x.to_le_bytes());
                        } else {
                            buf.extend_from_slice(&wav16(x).to_le_bytes());
                        }
                    }
                }
                w.write_all(&buf).map_err(e)?;
            }
            w.flush().map_err(e)
        }
        Format::Flac16 => {
            use flacenc::component::BitRepr;
            use flacenc::error::Verify;
            let mut inter = vec![0i32; frames * channels];
            for i in 0..frames {
                for c in 0..channels {
                    inter[i * channels + c] = flac16(data[c * frames + i]) as i32;
                }
            }
            let config = flacenc::config::Encoder::default().into_verified().map_err(|e| format!("{e:?}"))?;
            let source = flacenc::source::MemSource::from_samples(&inter, channels, 16, rate as usize);
            let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
                .map_err(|e| format!("{}: flac: {e:?}", path.display()))?;
            let mut sink = flacenc::bitsink::ByteSink::new();
            stream.write(&mut sink).map_err(|e| format!("{}: flac: {e:?}", path.display()))?;
            std::fs::write(path, sink.as_slice()).map_err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantisation_matches_libsndfile() {
        // Values and results measured from torchaudio 2.2.2's soundfile backend.
        let x = [0.0f32, 0.5 / 32768.0, 1.5 / 32768.0, -0.5 / 32768.0, -1.5 / 32768.0, 0.3, -0.3, 0.99, -0.99, 1.0, -1.0, 2.5 / 32767.0, 100.4 / 32768.0];
        let wav = [0i16, 0, 1, -1, -2, 9830, -9831, 32440, -32441, 32767, -32768, 2, 100];
        let flac = [0i16, 0, 2, 0, -2, 9830, -9830, 32440, -32440, 32767, -32768, 3, 100];
        for i in 0..x.len() {
            assert_eq!(wav16(x[i]), wav[i], "wav {}", x[i]);
            assert_eq!(flac16(x[i]), flac[i], "flac {}", x[i]);
        }
    }
}
