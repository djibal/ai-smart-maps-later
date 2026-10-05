//! The local update a device computes from its training pairs. One pass
//! of the margin ranking gradient on the A1-v5 device shape, 4 inputs,
//! two hidden layers of 32, one output, with a fixed learning rate. The
//! result is a weight delta for [`crate::aggregation::Round`]. It is not
//! a route, a coordinate, or a reporter key, and it leaves the device only
//! after the opt-in.

pub const INPUTS: usize = 4;
pub const HIDDEN: usize = 32;
pub const LAYERS: usize = 2;
/// 4·32 + 32 + 32·32 + 32 + 32·1 + 1.
pub const PARAMETERS: usize = INPUTS * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN + 1;
/// The A1-v5 training margin.
pub const MARGIN: f32 = 1.0;
/// The fixed step. One pass, one step on the mean gradient.
pub const LEARNING_RATE: f32 = 1e-3;

/// `[edge_count, weight_sum, hazard_sum, hazard_missing]`, the core's
/// feature order. The preferred route scores higher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub left: [f32; INPUTS],
    pub right: [f32; INPUTS],
    pub left_has_lower_cost: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum UpdateError {
    WrongShape,
    NoPairs,
}

/// Flat weights in layer order: W1 (32×4), b1, W2 (32×32), b2, W3 (1×32), b3.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights(pub Vec<f32>);

impl Weights {
    /// A deterministic small initialisation from one seed, for tests and
    /// for a device that has no artifact weights yet.
    pub fn seeded(seed: u64) -> Self {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.2 - 0.1
        };
        Self((0..PARAMETERS).map(|_| next()).collect())
    }

    pub fn is_well_formed(&self) -> bool {
        self.0.len() == PARAMETERS
    }
}

pub(crate) struct Layers<'a> {
    pub(crate) w1: &'a [f32],
    pub(crate) b1: &'a [f32],
    pub(crate) w2: &'a [f32],
    pub(crate) b2: &'a [f32],
    pub(crate) w3: &'a [f32],
    pub(crate) b3: f32,
}

pub(crate) fn split(weights: &[f32]) -> Option<Layers<'_>> {
    if weights.len() != PARAMETERS {
        return None;
    }
    let (w1, rest) = weights.split_at(HIDDEN * INPUTS);
    let (b1, rest) = rest.split_at(HIDDEN);
    let (w2, rest) = rest.split_at(HIDDEN * HIDDEN);
    let (b2, rest) = rest.split_at(HIDDEN);
    let (w3, rest) = rest.split_at(HIDDEN);
    Some(Layers {
        w1,
        b1,
        w2,
        b2,
        w3,
        b3: rest[0],
    })
}

struct Activations {
    z1: [f32; HIDDEN],
    h1: [f32; HIDDEN],
    z2: [f32; HIDDEN],
    h2: [f32; HIDDEN],
    score: f32,
}

fn forward(layers: &Layers<'_>, x: &[f32; INPUTS]) -> Activations {
    let mut z1 = [0f32; HIDDEN];
    let mut h1 = [0f32; HIDDEN];
    for (j, z) in z1.iter_mut().enumerate() {
        *z = layers.b1[j]
            + (0..INPUTS)
                .map(|i| layers.w1[j * INPUTS + i] * x[i])
                .sum::<f32>();
        h1[j] = z.max(0.0);
    }
    let mut z2 = [0f32; HIDDEN];
    let mut h2 = [0f32; HIDDEN];
    for (k, z) in z2.iter_mut().enumerate() {
        *z = layers.b2[k]
            + (0..HIDDEN)
                .map(|j| layers.w2[k * HIDDEN + j] * h1[j])
                .sum::<f32>();
        h2[k] = z.max(0.0);
    }
    let score = layers.b3 + (0..HIDDEN).map(|k| layers.w3[k] * h2[k]).sum::<f32>();
    Activations {
        z1,
        h1,
        z2,
        h2,
        score,
    }
}

/// The scorer's output for one feature vector. Higher is preferred.
pub fn score(weights: &Weights, x: &[f32; INPUTS]) -> Result<f32, UpdateError> {
    let layers = split(&weights.0).ok_or(UpdateError::WrongShape)?;
    Ok(forward(&layers, x).score)
}

/// Adds `upstream · ∂score/∂θ` for one input into `grad`.
fn backward(
    layers: &Layers<'_>,
    x: &[f32; INPUTS],
    acts: &Activations,
    upstream: f32,
    grad: &mut [f32],
) {
    let (g_w1, rest) = grad.split_at_mut(HIDDEN * INPUTS);
    let (g_b1, rest) = rest.split_at_mut(HIDDEN);
    let (g_w2, rest) = rest.split_at_mut(HIDDEN * HIDDEN);
    let (g_b2, rest) = rest.split_at_mut(HIDDEN);
    let (g_w3, g_b3) = rest.split_at_mut(HIDDEN);

    g_b3[0] += upstream;
    let mut d_h2 = [0f32; HIDDEN];
    for k in 0..HIDDEN {
        g_w3[k] += upstream * acts.h2[k];
        d_h2[k] = upstream * layers.w3[k];
    }
    let mut d_h1 = [0f32; HIDDEN];
    for k in 0..HIDDEN {
        let d_z2 = if acts.z2[k] > 0.0 { d_h2[k] } else { 0.0 };
        g_b2[k] += d_z2;
        for j in 0..HIDDEN {
            g_w2[k * HIDDEN + j] += d_z2 * acts.h1[j];
            d_h1[j] += d_z2 * layers.w2[k * HIDDEN + j];
        }
    }
    for j in 0..HIDDEN {
        let d_z1 = if acts.z1[j] > 0.0 { d_h1[j] } else { 0.0 };
        g_b1[j] += d_z1;
        for i in 0..INPUTS {
            g_w1[j * INPUTS + i] += d_z1 * x[i];
        }
    }
}

/// Mean margin ranking loss over the pairs:
/// `max(0, margin − y·(score(left) − score(right)))`, `y = +1` when the
/// left route has the lower cost.
pub fn loss(weights: &Weights, pairs: &[Pair]) -> Result<f32, UpdateError> {
    let layers = split(&weights.0).ok_or(UpdateError::WrongShape)?;
    if pairs.is_empty() {
        return Err(UpdateError::NoPairs);
    }
    let total: f32 = pairs
        .iter()
        .map(|pair| {
            let y = if pair.left_has_lower_cost { 1.0 } else { -1.0 };
            let gap = forward(&layers, &pair.left).score - forward(&layers, &pair.right).score;
            (MARGIN - y * gap).max(0.0)
        })
        .sum();
    Ok(total / pairs.len() as f32)
}

/// One pass over the pairs, one step of size [`LEARNING_RATE`] against the
/// mean gradient. Returns the delta, not the new weights. The same weights
/// and pairs always give the same delta.
pub fn local_update(weights: &Weights, pairs: &[Pair]) -> Result<Vec<f32>, UpdateError> {
    let layers = split(&weights.0).ok_or(UpdateError::WrongShape)?;
    if pairs.is_empty() {
        return Err(UpdateError::NoPairs);
    }
    let mut grad = vec![0f32; PARAMETERS];
    for pair in pairs {
        let y = if pair.left_has_lower_cost { 1.0 } else { -1.0 };
        let left = forward(&layers, &pair.left);
        let right = forward(&layers, &pair.right);
        if MARGIN - y * (left.score - right.score) <= 0.0 {
            continue;
        }
        backward(&layers, &pair.left, &left, -y, &mut grad);
        backward(&layers, &pair.right, &right, y, &mut grad);
    }
    let scale = -LEARNING_RATE / pairs.len() as f32;
    Ok(grad.into_iter().map(|g| g * scale).collect())
}
