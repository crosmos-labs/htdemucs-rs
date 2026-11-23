//! The streaming stem writer gives the same bytes as clipping the whole stem and writing it at
//! once (audio::write), in every format and clip mode, however the frames are pushed.

use htdemucs::audio::{self, Clip, Format, StemWriter};

fn clip_whole(x: &mut [f32], clip: Clip) {
    match clip {
        Clip::Rescale => {
            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = (1.01f32 * peak).max(1.0);
            x.iter_mut().for_each(|v| *v /= d);
        }
        Clip::Clamp => x.iter_mut().for_each(|v| *v = v.clamp(-0.99, 0.99)),
        Clip::None => {}
    }
}

#[test]
fn streamed_stems_equal_whole_writes() {
    let dir = std::env::temp_dir().join(format!("htdemucs-writer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Two channels, a length that isn't a whole number of FLAC blocks, peaks above 1 (rescale
    // and clamp both act) - and a quiet one (rescale divides by 1).
    let frames = 3 * 4096 * 5 + 1234;
    for (k, amp) in [1.7f32, 0.4].into_iter().enumerate() {
        let data: Vec<f32> = (0..2 * frames).map(|i| ((i as f32) * 0.00731).sin() * amp * if i % 977 == 0 { 1.3 } else { 1.0 }).collect();
        for fmt in [Format::Wav16, Format::WavF32, Format::Flac16] {
            for clip in [Clip::Rescale, Clip::Clamp, Clip::None] {
                let mut x = data.clone();
                clip_whole(&mut x, clip);
                let a = dir.join(format!("a{k}.{}", fmt.ext()));
                audio::write(&a, &x, 2, 44100, fmt).unwrap();
                let b = dir.join(format!("b{k}.{}", fmt.ext()));
                let mut w = StemWriter::create(&b, 2, frames, 44100, fmt, clip).unwrap();
                let (l, r) = data.split_at(frames);
                let mut i = 0;
                let mut n = 1;
                while i < frames {
                    let j = (i + n * 3001).min(frames);
                    w.push(&[&l[i..j], &r[i..j]]).unwrap();
                    i = j;
                    n = n % 9 + 1;
                }
                w.finish().unwrap();
                assert!(std::fs::read(&a).unwrap() == std::fs::read(&b).unwrap(), "{fmt:?} {clip:?} amp {amp}: bytes differ");
                assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "temporary files left behind");
                std::fs::remove_file(&a).unwrap();
                std::fs::remove_file(&b).unwrap();
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
