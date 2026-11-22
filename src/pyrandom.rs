//! CPython's `random` module (3.11), as far as Demucs uses it: MT19937 seeded by `random.seed(n)`
//! for an int n, `getrandbits(k)`, `randrange(n)` and `randint(a, b)`.
//!
//! Why it matters: `demucs.apply.apply_model` draws each shift's offset with
//! `random.randint(0, max_shift)`, and every segment forward in between draws
//! `random.randrange(1)` (the transformer's `sin_random_shift + 1`). randrange(1) always returns 0
//! but still consumes generator output (getrandbits(1), repeated until it returns 0), so the
//! second shift's offset depends on how many segments the first pass had.

pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    /// `random.seed(n)` for an int n: init_by_array over the 32-bit words of |n|, least
    /// significant first ([0] for n = 0).
    pub fn new(seed: i128) -> Self {
        let mut n = seed.unsigned_abs();
        let mut key = vec![];
        while n > 0 {
            key.push(n as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = PyRandom { mt: [0; 624], idx: 624 };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let n = 624;
        for _ in 0..n.max(key.len()) {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..n - 1 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn genrand_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & UPPER) | (self.mt[(k + 1) % 624] & LOWER);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for 0 < k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        assert!((1..=32).contains(&k));
        self.genrand_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)`: rejection sampling on n.bit_length() bits.
    fn randbelow(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `randrange(stop)`, stop > 0.
    pub fn randrange(&mut self, stop: u32) -> u32 {
        assert!(stop > 0);
        self.randbelow(stop)
    }

    /// `randint(a, b)` = randrange(a, b + 1).
    pub fn randint(&mut self, a: u32, b: u32) -> u32 {
        a + self.randbelow(b - a + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    /// Logged from CPython 3.11.17: (seed, first four getrandbits(32), [randint(0, 22050),
    /// then after 40 x randrange(1): randint(0, 22050), randint(0, 22050)],
    /// [getrandbits(1), getrandbits(15), getrandbits(31), getrandbits(32)] from a fresh seed).
    const CASES: &[(i128, [u32; 4], [u32; 3], [u32; 4])] = &[
        (0, [3626764237, 1654615998, 3255389356, 3823568514], [12623, 7818, 4669], [1, 12623, 1627694678, 3823568514]),
        (1, [577090037, 2444712010, 3639700191, 3445702192], [4402, 20622, 9712], [0, 18651, 1819850095, 3445702192]),
        (42, [2746317213, 478163327, 107420369, 3184935163], [20952, 18089, 9606], [1, 3648, 53710184, 3184935163]),
        (12345, [1789368711, 3146859322, 43676229, 3522623596], [13651, 1318, 7612], [0, 24008, 21838114, 3522623596]),
        ((1 << 40) + 7, [2635837658, 3209733218, 3500038837, 4028389051], [20109, 18434, 13355], [1, 24488, 1750019418, 4028389051]),
        (-5, [2675342405, 1097127993, 3185950873, 1539898300], [20411, 706, 11835], [1, 8370, 1592975436, 1539898300]),
    ];

    #[test]
    fn matches_cpython() {
        for &(seed, words, seq, bits) in CASES {
            let mut r = PyRandom::new(seed);
            let w: Vec<u32> = (0..4).map(|_| r.getrandbits(32)).collect();
            assert_eq!(w, words, "seed {seed} words");
            let mut r = PyRandom::new(seed);
            let mut s = vec![r.randint(0, 22050)];
            for _ in 0..40 {
                assert_eq!(r.randrange(1), 0);
            }
            s.push(r.randint(0, 22050));
            s.push(r.randint(0, 22050));
            assert_eq!(s, seq, "seed {seed} randint/randrange");
            let mut r = PyRandom::new(seed);
            let b: Vec<u32> = [1, 15, 31, 32].iter().map(|&k| r.getrandbits(k)).collect();
            assert_eq!(b, bits, "seed {seed} getrandbits");
        }
    }

    /// The sequence a seeded demucs run draws on a 20 s clip (logged from demucs 4.0.1 itself):
    /// shift offsets 12623 and 13268, with 4 segment forwards in between.
    #[test]
    fn demucs_20s_clip() {
        let mut r = PyRandom::new(0);
        assert_eq!(r.randint(0, 22050), 12623);
        for _ in 0..4 {
            r.randrange(1);
        }
        assert_eq!(r.randint(0, 22050), 13268);
    }
}
