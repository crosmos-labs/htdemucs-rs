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
    // MP4: edit list and AAC config (symphonia handles neither SBR/PS nor edit lists).
    let mp4_info = mp4_audio(path, params.sample_rate.unwrap_or(0))?;
    if let Some(asc) = mp4_info.as_ref().and_then(|m| m.asc.as_deref()) {
        let core_rate = params.sample_rate.unwrap_or(0);
        if asc_is_he_aac(asc) || (core_rate > 0 && core_rate <= 24_000) {
            return Err(format!(
                "{}: HE-AAC (AAC with SBR / PS{}) is not supported: only its {core_rate} Hz core would be decoded, giving wrong, late stems. Decode it with ffmpeg first: ffmpeg -i '{}' -c:a pcm_f32le track.wav",
                path.display(),
                if asc_is_he_aac(asc) { "" } else { ", implied by a sample rate of 24 kHz or less" },
                path.display()
            ));
        }
    }
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
    if mp4_info.as_ref().is_some_and(|m| m.edit.is_some()) && Some(rate) != params.sample_rate {
        return Err(format!("{}: decoded at {rate} Hz but the container says {:?}: not supported", path.display(), params.sample_rate));
    }
    let mut frames = inter.len() / channels;
    let mut start = 0;
    if let Some(mp4) = mp4_info {
        if let Some((skip, keep)) = mp4.edit {
            start = skip.min(frames);
            frames = keep.map_or(frames - start, |k| k.min(frames - start));
        }
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

/// What we need from an MP4/M4A's sound track that symphonia doesn't give us.
#[derive(Debug, Default, PartialEq)]
struct Mp4Audio {
    /// (samples to skip, samples to keep) from the edit list, as ffmpeg applies it.
    edit: Option<(usize, Option<usize>)>,
    /// The track is MPEG-4 audio (AAC): its AudioSpecificConfig, if present.
    asc: Option<Vec<u8>>,
}

/// Reads the sound track's edit list and AAC config from an MP4/M4A. Ok(None) if the file is not
/// an MP4. Every read is bounds-checked: a malformed file gives an error, never a panic.
fn mp4_audio(path: &Path, rate: u32) -> Result<Option<Mp4Audio>, String> {
    let bad = |what: &str| format!("{}: malformed MP4 ({what})", path.display());
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
        f.read_exact(&mut h[..8]).map_err(|_| bad("box header"))?;
        let mut size = u32::from_be_bytes(h[..4].try_into().unwrap()) as u64;
        let mut hdr = 8u64;
        if size == 1 {
            f.read_exact(&mut h[8..16]).map_err(|_| bad("box header"))?;
            size = u64::from_be_bytes(h[8..16].try_into().unwrap());
            hdr = 16;
        } else if size == 0 {
            size = len - pos;
        }
        if size < hdr || size > len - pos {
            return Err(bad("box size"));
        }
        if &h[4..8] == b"moov" {
            let mut b = vec![0u8; (size - hdr) as usize];
            f.read_exact(&mut b).map_err(|_| bad("moov"))?;
            moov = Some(b);
            break;
        }
        pos += size;
    }
    let Some(moov) = moov else { return Ok(None) };
    let movie_ts = child(&moov, b"mvhd").and_then(timescale).unwrap_or(0);
    let mut off = 0;
    while let Some((typ, body, next)) = next_box(&moov, off) {
        off = next;
        if typ != *b"trak" {
            continue;
        }
        let Some(mdia) = child(body, b"mdia") else { continue };
        let is_sound = child(mdia, b"hdlr").is_some_and(|h| h.get(8..12) == Some(b"soun"));
        if !is_sound {
            continue;
        }
        let mut out = Mp4Audio::default();
        out.asc = child(mdia, b"minf")
            .and_then(|m| child(m, b"stbl"))
            .and_then(|s| child(s, b"stsd"))
            .and_then(|s| s.get(8..)) // full box header + entry count
            .and_then(|e| child(e, b"mp4a"))
            .and_then(|a| a.get(28..)) // the audio sample entry's fixed fields
            .and_then(|a| child(a, b"esds"))
            .and_then(|e| e.get(4..))
            .and_then(audio_specific_config);
        let media_ts = child(mdia, b"mdhd").and_then(timescale).unwrap_or(0);
        let Some(elst) = child(body, b"edts").and_then(|e| child(e, b"elst")) else { return Ok(Some(out)) };
        let rd = |a: usize, n: usize| elst.get(a..a + n).ok_or_else(|| bad("elst"));
        let version = rd(0, 1)?[0];
        let count = u32::from_be_bytes(rd(4, 4)?.try_into().unwrap());
        let mut p = 8;
        let mut skip = 0i64;
        let mut keep_movie = 0u64;
        for i in 0..count {
            let (dur, mt) = if version == 1 {
                let d = u64::from_be_bytes(rd(p, 8)?.try_into().unwrap());
                let m = i64::from_be_bytes(rd(p + 8, 8)?.try_into().unwrap());
                p += 20;
                (d, m)
            } else {
                let d = u32::from_be_bytes(rd(p, 4)?.try_into().unwrap()) as u64;
                let m = i32::from_be_bytes(rd(p + 4, 4)?.try_into().unwrap()) as i64;
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
            keep_movie = keep_movie.saturating_add(dur);
        }
        if media_ts > 0 && movie_ts > 0 {
            let to_samples = |v: u128, ts: u64| ((v * rate as u128 + ts as u128 / 2) / ts as u128) as usize;
            let s = to_samples(skip.max(0) as u128, media_ts);
            let keep = (keep_movie > 0).then(|| to_samples(keep_movie as u128, movie_ts));
            out.edit = Some((s, keep));
        }
        return Ok(Some(out));
    }
    Ok(None)
}

/// The AudioSpecificConfig inside an esds box's ES_Descriptor (tags 0x03 > 0x04 > 0x05), for
/// MPEG-4 audio (objectTypeIndication 0x40) only.
fn audio_specific_config(esds: &[u8]) -> Option<Vec<u8>> {
    // A descriptor: tag, then a length in 7-bit groups (high bit = more), then the body.
    fn desc(b: &[u8]) -> Option<(u8, &[u8])> {
        let tag = *b.first()?;
        let mut n = 0usize;
        let mut i = 1;
        loop {
            let c = *b.get(i)?;
            n = (n << 7) | (c & 0x7f) as usize;
            i += 1;
            if c & 0x80 == 0 || i > 4 {
                break;
            }
        }
        Some((tag, b.get(i..i + n)?))
    }
    let (t, es) = desc(esds)?;
    if t != 0x03 {
        return None;
    }
    let flags = *es.get(2)?;
    let mut i = 3;
    if flags & 0x80 != 0 {
        i += 2; // dependsOn_ES_ID
    }
    if flags & 0x40 != 0 {
        i += 1 + *es.get(i)? as usize; // URL
    }
    if flags & 0x20 != 0 {
        i += 2; // OCR_ES_Id
    }
    let (t, dc) = desc(es.get(i..)?)?;
    if t != 0x04 || *dc.first()? != 0x40 {
        return None;
    }
    let (t, asc) = desc(dc.get(13..)?)?;
    (t == 0x05).then(|| asc.to_vec())
}

/// Does an AudioSpecificConfig describe HE-AAC (SBR) or HE-AACv2 (PS)? Explicit signalling
/// (audio object type 5 or 29) or the backward-compatible kind Apple writes: an AAC-LC config
/// followed by sync extension 0x2b7 with object type 5 and sbrPresentFlag set.
pub fn asc_is_he_aac(asc: &[u8]) -> bool {
    struct Bits<'a> {
        b: &'a [u8],
        pos: usize,
    }
    impl Bits<'_> {
        fn get(&mut self, n: usize) -> Option<u32> {
            let mut v = 0u32;
            for _ in 0..n {
                let byte = *self.b.get(self.pos / 8)?;
                v = (v << 1) | ((byte >> (7 - self.pos % 8)) & 1) as u32;
                self.pos += 1;
            }
            Some(v)
        }
        fn left(&self) -> usize {
            (self.b.len() * 8).saturating_sub(self.pos)
        }
    }
    let mut r = Bits { b: asc, pos: 0 };
    let aot = |r: &mut Bits| -> Option<u32> {
        let a = r.get(5)?;
        if a == 31 { Some(32 + r.get(6)?) } else { Some(a) }
    };
    let check = |r: &mut Bits| -> Option<bool> {
        let a = aot(r)?;
        if a == 5 || a == 29 {
            return Some(true);
        }
        if r.get(4)? == 15 {
            r.get(24)?;
        }
        let channels = r.get(4)?;
        if !matches!(a, 1 | 2 | 3 | 4 | 6 | 7) || channels == 0 {
            return Some(false); // not a plain GA config we can walk past
        }
        // GASpecificConfig: frameLengthFlag, dependsOnCoreCoder (+ coreCoderDelay), extensionFlag
        r.get(1)?;
        if r.get(1)? == 1 {
            r.get(14)?;
        }
        r.get(1)?;
        if r.left() >= 16 && r.get(11)? == 0x2b7 && aot(r)? == 5 {
            return Some(r.get(1)? == 1);
        }
        Some(false)
    };
    check(&mut r).unwrap_or(false)
}

fn timescale(full_box: &[u8]) -> Option<u64> {
    // mvhd / mdhd: version(1) flags(3), then v0: ctime(4) mtime(4) timescale(4); v1: 8, 8, 4.
    let at = if *full_box.first()? == 1 { 20 } else { 12 };
    Some(u32::from_be_bytes(full_box.get(at..at + 4)?.try_into().ok()?) as u64)
}

fn next_box(b: &[u8], off: usize) -> Option<([u8; 4], &[u8], usize)> {
    let mut size = u32::from_be_bytes(b.get(off..off + 4)?.try_into().ok()?) as usize;
    let typ: [u8; 4] = b.get(off + 4..off + 8)?.try_into().ok()?;
    let mut hdr = 8;
    if size == 1 {
        size = usize::try_from(u64::from_be_bytes(b.get(off + 8..off + 16)?.try_into().ok()?)).ok()?;
        hdr = 16;
    } else if size == 0 {
        size = b.len() - off;
    }
    if size < hdr || off.checked_add(size)? > b.len() {
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
            let config = flacenc::config::Encoder::default().into_verified().map_err(|e| format!("{e:?}"))?;
            let source = PlanarSource { data, channels, frames, rate: rate as usize, pos: 0, buf: vec![] };
            let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
                .map_err(|e| format!("{}: flac: {e:?}", path.display()))?;
            let mut sink = flacenc::bitsink::ByteSink::new();
            stream.write(&mut sink).map_err(|e| format!("{}: flac: {e:?}", path.display()))?;
            std::fs::write(path, sink.as_slice()).map_err(e)
        }
    }
}

/// Planar f32 samples as a flacenc source, quantised block by block (no whole-track i32 copy).
struct PlanarSource<'a> {
    data: &'a [f32],
    channels: usize,
    frames: usize,
    rate: usize,
    pos: usize,
    buf: Vec<i32>,
}

impl flacenc::source::Source for PlanarSource<'_> {
    fn channels(&self) -> usize {
        self.channels
    }
    fn bits_per_sample(&self) -> usize {
        16
    }
    fn sample_rate(&self) -> usize {
        self.rate
    }
    fn read_samples<F: flacenc::source::Fill>(&mut self, block_size: usize, dest: &mut F) -> Result<usize, flacenc::error::SourceError> {
        let n = block_size.min(self.frames - self.pos);
        self.buf.clear();
        for i in self.pos..self.pos + n {
            for c in 0..self.channels {
                self.buf.push(flac16(self.data[c * self.frames + i]) as i32);
            }
        }
        dest.fill_interleaved(&self.buf)?;
        self.pos += n;
        Ok(n)
    }
    fn len_hint(&self) -> Option<usize> {
        Some(self.frames)
    }
}

#[cfg(test)]
mod tests {
    fn bx(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(typ);
        v.extend_from_slice(body);
        v
    }

    /// A minimal MP4: ftyp + moov(mvhd, trak(edts(elst), mdia(mdhd, hdlr soun))).
    fn mp4(elst: &[u8]) -> Vec<u8> {
        let mut mvhd = vec![0u8; 12];
        mvhd.extend_from_slice(&1000u32.to_be_bytes());
        let mut mdhd = vec![0u8; 12];
        mdhd.extend_from_slice(&44100u32.to_be_bytes());
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(b"soun");
        let mdia = [bx(b"mdhd", &mdhd), bx(b"hdlr", &hdlr)].concat();
        let trak = [bx(b"edts", &bx(b"elst", elst)), bx(b"mdia", &mdia)].concat();
        let moov = [bx(b"mvhd", &mvhd), bx(b"trak", &trak)].concat();
        [bx(b"ftyp", b"M4A \0\0\0\0"), bx(b"moov", &moov)].concat()
    }

    fn elst_v0(entries: &[(u32, i32)], count: u32) -> Vec<u8> {
        let mut e = vec![0u8; 4];
        e.extend_from_slice(&count.to_be_bytes());
        for &(d, m) in entries {
            e.extend_from_slice(&d.to_be_bytes());
            e.extend_from_slice(&m.to_be_bytes());
            e.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        }
        e
    }

    fn parse(bytes: &[u8]) -> Result<Option<Mp4Audio>, String> {
        let dir = std::env::temp_dir().join(format!("htdemucs-mp4-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::write(&dir, bytes).unwrap();
        let r = mp4_audio(&dir, 44100);
        std::fs::remove_file(&dir).ok();
        r
    }

    #[test]
    fn mp4_edit_list() {
        // 1 s kept (movie timescale 1000), 2112 priming samples skipped.
        let good = mp4(&elst_v0(&[(1000, 2112)], 1));
        assert_eq!(parse(&good).unwrap().unwrap().edit, Some((2112, Some(44100))));
    }

    #[test]
    fn malformed_mp4_is_an_error_not_a_panic() {
        // A count claiming more entries than there are bytes.
        assert!(parse(&mp4(&elst_v0(&[(1000, 2112)], 1_000_000))).is_err());
        // Every truncation of a valid file: an error or a partial answer, never a panic.
        let good = mp4(&elst_v0(&[(1000, 2112)], 1));
        for n in 0..good.len() {
            let _ = parse(&good[..n]);
        }
        // Every single-byte corruption of the box headers' size fields.
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] = 0xff;
            let _ = parse(&b);
        }
    }

    #[test]
    fn he_aac_configs() {
        // AudioSpecificConfigs as Apple's afconvert writes them (AAC-LC core + sync extension).
        assert!(asc_is_he_aac(&[0x13, 0x90, 0x56, 0xe5, 0xa0])); // HE-AAC, 22.05 kHz core, stereo
        assert!(asc_is_he_aac(&[0x13, 0x88, 0x56, 0xe5, 0xa5, 0x48, 0x80])); // HE-AACv2 (PS)
        assert!(!asc_is_he_aac(&[0x12, 0x10])); // AAC-LC 44.1 kHz stereo
        assert!(!asc_is_he_aac(&[0x11, 0x90])); // AAC-LC 48 kHz stereo
        // Explicit signalling: object type 5 (SBR) / 29 (PS).
        assert!(asc_is_he_aac(&[0x2b, 0x92, 0x08, 0x00]));
        assert!(asc_is_he_aac(&[0xeb, 0x92, 0x08, 0x00]));
        // Garbage never panics.
        for n in 0..8 {
            let _ = asc_is_he_aac(&[0xff; 8][..n]);
        }
    }

    #[test]
    fn he_aac_files_are_refused() {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        for f in ["he-aac-1s.m4a", "he-aac-v2-1s.m4a"] {
            let e = decode(&data.join(f)).err().unwrap_or_else(|| panic!("{f} decoded"));
            assert!(e.contains("HE-AAC") && e.contains("ffmpeg"), "{e}");
        }
        let lc = decode(&data.join("aac-lc-1s.m4a")).unwrap();
        assert_eq!((lc.rate, lc.channels), (44100, 2));
    }

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
