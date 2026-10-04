//! Device-local canary. The draw is a local number, not an identity.
//! A candidate more than 5 percent off the cloud reference does not enter.
//! A candidate more than 3 percentage points behind the previous version
//! stays off the device.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub previous_id: String,
    pub candidate_id: String,
    pub passed: bool,
}

/// About 1 percent of draws. `draw % 100 == 0` is the candidate bucket.
pub fn in_bucket(draw: u32) -> bool {
    draw % 100 == 0
}

pub fn assigned_id(previous_id: &str, candidate_id: &str, draw: u32) -> String {
    if in_bucket(draw) {
        candidate_id.to_string()
    } else {
        previous_id.to_string()
    }
}

/// True when the device mean is at or above the reference mean minus 3 points.
pub fn within_three_points(candidate_mean: f64, reference_mean: f64) -> bool {
    candidate_mean >= reference_mean - 3.0
}

/// True unless the device mean is more than 5 percent off the cloud mean.
pub fn within_five_percent(device_mean: f64, cloud_mean: f64) -> bool {
    if cloud_mean <= 0.0 {
        return false;
    }
    (cloud_mean - device_mean).abs() / cloud_mean <= 0.05
}

pub fn decide(
    previous_id: &str,
    candidate_id: &str,
    candidate_mean: f64,
    previous_mean: f64,
    cloud_mean: f64,
) -> Record {
    let passed = within_five_percent(candidate_mean, cloud_mean)
        && within_three_points(candidate_mean, previous_mean);
    Record {
        previous_id: previous_id.to_string(),
        candidate_id: candidate_id.to_string(),
        passed,
    }
}

pub fn active_id(record: &Record) -> &str {
    if record.passed {
        &record.candidate_id
    } else {
        &record.previous_id
    }
}
