//! Federated aggregation round. Krum is the primary aggregator.
//! Trimmed mean runs only when `trim` is at least `f / n`.
//! The proven setting is `f = 3`, `m = 3`, and at most 10 devices.
//! A failed round returns an error and leaves the active scorer as it is.

pub const PROVEN_F: usize = 3;
pub const PROVEN_M: usize = 3;
pub const PROVEN_COHORT: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub struct Upload {
    pub opted_in: bool,
    pub delta: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    Krum { m: usize },
    TrimmedMean { trim: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActiveScorer {
    pub id: String,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub previous_id: String,
    pub weights: Vec<f32>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RoundError {
    NotOptedIn,
    Unmeasured,
    TooFew,
    TrimTooSmall,
    Mismatch,
}

#[derive(Clone, Debug)]
pub struct Round {
    f: usize,
    deltas: Vec<Vec<f32>>,
}

impl Round {
    /// `f` is declared before any upload arrives.
    pub fn new(f: usize) -> Self {
        Self {
            f,
            deltas: Vec::new(),
        }
    }

    pub fn f(&self) -> usize {
        self.f
    }

    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Keeps a delta only from a device that opted in.
    pub fn accept(&mut self, upload: Upload) -> Result<(), RoundError> {
        if !upload.opted_in {
            return Err(RoundError::NotOptedIn);
        }
        self.deltas.push(upload.delta);
        Ok(())
    }

    /// Produces a candidate whose `previous_id` is the active scorer.
    /// The active scorer is not changed by this call.
    pub fn aggregate(
        &self,
        method: Method,
        active: &ActiveScorer,
        candidate_id: &str,
    ) -> Result<Candidate, RoundError> {
        let delta = match method {
            Method::Krum { m } => krum(&self.deltas, self.f, m)?,
            Method::TrimmedMean { trim } => trimmed_mean(&self.deltas, self.f, trim)?,
        };
        if delta.len() != active.weights.len() {
            return Err(RoundError::Mismatch);
        }
        let weights = active
            .weights
            .iter()
            .zip(&delta)
            .map(|(weight, change)| weight + change)
            .collect();
        Ok(Candidate {
            id: candidate_id.to_string(),
            previous_id: active.id.clone(),
            weights,
        })
    }
}

fn same_length(deltas: &[Vec<f32>]) -> Result<usize, RoundError> {
    let first = deltas.first().ok_or(RoundError::TooFew)?;
    if deltas.iter().any(|delta| delta.len() != first.len()) {
        return Err(RoundError::Mismatch);
    }
    Ok(first.len())
}

fn proven(f: usize, n: usize) -> Result<(), RoundError> {
    if f != PROVEN_F || n > PROVEN_COHORT {
        return Err(RoundError::Unmeasured);
    }
    Ok(())
}

/// Multi-Krum. Each delta is scored by the sum of squared distances to its
/// `n - f - 2` nearest neighbours. The `m` lowest scores are averaged.
/// Needs at least `2f + 3` deltas. There is no fallback to a plain mean.
pub fn krum(deltas: &[Vec<f32>], f: usize, m: usize) -> Result<Vec<f32>, RoundError> {
    let n = deltas.len();
    proven(f, n)?;
    if m != PROVEN_M {
        return Err(RoundError::Unmeasured);
    }
    if n < 2 * f + 3 {
        return Err(RoundError::TooFew);
    }
    let width = same_length(deltas)?;
    let neighbours = n - f - 2;
    let mut scored: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let mut distances: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| squared_distance(&deltas[i], &deltas[j]))
                .collect();
            distances.sort_by(|a, b| a.total_cmp(b));
            (distances.iter().take(neighbours).sum(), i)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let chosen: Vec<&Vec<f32>> = scored.iter().take(m).map(|(_, i)| &deltas[*i]).collect();
    Ok(mean(&chosen, width))
}

/// Coordinate-wise trimmed mean. `trim` must be at least `f / n`, so the
/// `floor(trim * n)` values dropped at each end cover every malicious delta.
pub fn trimmed_mean(deltas: &[Vec<f32>], f: usize, trim: f32) -> Result<Vec<f32>, RoundError> {
    let n = deltas.len();
    proven(f, n)?;
    if n == 0 {
        return Err(RoundError::TooFew);
    }
    if trim < f as f32 / n as f32 {
        return Err(RoundError::TrimTooSmall);
    }
    let width = same_length(deltas)?;
    let drop = (trim * n as f32).floor() as usize;
    if 2 * drop >= n {
        return Err(RoundError::TooFew);
    }
    let mut out = Vec::with_capacity(width);
    for column in 0..width {
        let mut values: Vec<f32> = deltas.iter().map(|delta| delta[column]).collect();
        values.sort_by(|a, b| a.total_cmp(b));
        let kept = &values[drop..n - drop];
        out.push(kept.iter().sum::<f32>() / kept.len() as f32);
    }
    Ok(out)
}

fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| (a - b) * (a - b)).sum()
}

fn mean(deltas: &[&Vec<f32>], width: usize) -> Vec<f32> {
    let count = deltas.len() as f32;
    (0..width)
        .map(|column| deltas.iter().map(|delta| delta[column]).sum::<f32>() / count)
        .collect()
}
