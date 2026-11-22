# htdemucs-rs

Music stem separation (HTDemucs v4) in Rust: the same model and weights as Meta's Demucs, without
Python or PyTorch. Built for Fretboard's song pipeline, where it replaces `demucs` (htdemucs and
htdemucs_6s, 2 seeded shifts).

Priorities, in order: quality (match PyTorch Demucs), speed, build size.

Status: in progress.
