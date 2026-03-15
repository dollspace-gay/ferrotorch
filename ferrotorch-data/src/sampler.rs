/// A sampler produces a sequence of indices for a `DataLoader` to fetch.
pub trait Sampler: Send + Sync {
    /// Return indices for one epoch.
    fn indices(&self, epoch: usize) -> Vec<usize>;

    /// Total number of samples.
    fn len(&self) -> usize;
}

/// Yields indices in order: 0, 1, 2, ..., n-1.
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    fn indices(&self, _epoch: usize) -> Vec<usize> {
        (0..self.size).collect()
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Yields indices in a random permutation, seeded by epoch for reproducibility.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    size: usize,
    seed: u64,
}

impl RandomSampler {
    pub fn new(size: usize, seed: u64) -> Self {
        Self { size, seed }
    }
}

impl Sampler for RandomSampler {
    fn indices(&self, epoch: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.size).collect();
        // Fisher-Yates shuffle with deterministic seed.
        let mut state = self.seed ^ (epoch as u64).wrapping_mul(0x9e3779b97f4a7c15);
        if state == 0 {
            state = 0xdeadbeefcafe;
        }
        for i in (1..indices.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            indices.swap(i, j);
        }
        indices
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let s = SequentialSampler::new(5);
        assert_eq!(s.indices(0), vec![0, 1, 2, 3, 4]);
        assert_eq!(s.indices(1), vec![0, 1, 2, 3, 4]); // Same every epoch.
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_random_sampler_permutation() {
        let s = RandomSampler::new(10, 42);
        let idx = s.indices(0);
        assert_eq!(idx.len(), 10);
        // Contains all indices.
        let mut sorted = idx.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_random_sampler_reproducible() {
        let s = RandomSampler::new(100, 42);
        let a = s.indices(0);
        let b = s.indices(0);
        assert_eq!(a, b); // Same seed+epoch = same order.
    }

    #[test]
    fn test_random_sampler_different_epochs() {
        let s = RandomSampler::new(20, 42);
        let a = s.indices(0);
        let b = s.indices(1);
        assert_ne!(a, b); // Different epochs = different order.
    }

    #[test]
    fn test_random_sampler_shuffled() {
        let s = RandomSampler::new(100, 42);
        let idx = s.indices(0);
        let sequential: Vec<usize> = (0..100).collect();
        assert_ne!(idx, sequential); // Should be shuffled.
    }
}
