//! Hazard broadcast rules. A message is not kept.
//! The receiver writes a local report, or drops the broadcast.

use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast {
    pub kind: String,
    pub edge_id: String,
    pub observed_at: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalReport {
    pub kind: String,
    pub edge_id: String,
    pub observed_at: String,
}

#[derive(Clone, Debug, Default)]
pub struct Receiver {
    clock: String,
    seen: HashSet<(String, String, String)>,
    reports: Vec<LocalReport>,
}

impl Receiver {
    pub fn new(clock: &str) -> Self {
        Self {
            clock: clock.to_string(),
            seen: HashSet::new(),
            reports: Vec::new(),
        }
    }

    pub fn reports(&self) -> &[LocalReport] {
        &self.reports
    }

    /// Accepts a broadcast that is at most 2 seconds behind the clock and
    /// has not been accepted before. Anything else leaves the reports as they were.
    pub fn accept(&mut self, message: Broadcast) -> bool {
        let Some(age) = seconds_behind(&self.clock, &message.observed_at) else {
            return false;
        };
        if age > 2 {
            return false;
        }
        let key = (
            message.kind.clone(),
            message.edge_id.clone(),
            message.observed_at.clone(),
        );
        if !self.seen.insert(key) {
            return false;
        }
        self.reports.push(LocalReport {
            kind: message.kind,
            edge_id: message.edge_id,
            observed_at: message.observed_at,
        });
        true
    }
}

fn seconds_behind(clock: &str, observed_at: &str) -> Option<i64> {
    Some(unix(clock)? - unix(observed_at)?)
}

fn unix(stamp: &str) -> Option<i64> {
    let bytes = stamp.as_bytes();
    if bytes.len() != 20 || bytes[10] != b'T' || bytes[19] != b'Z' {
        return None;
    }
    let year: i64 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
    let month: u32 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
    let day: u32 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
    let hour: u32 = std::str::from_utf8(&bytes[11..13]).ok()?.parse().ok()?;
    let minute: u32 = std::str::from_utf8(&bytes[14..16]).ok()?.parse().ok()?;
    let second: u32 = std::str::from_utf8(&bytes[17..19]).ok()?.parse().ok()?;
    if !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let year = year - (month <= 2) as i64;
    let era = year.div_euclid(400);
    let yoe = year.rem_euclid(400) as u32;
    let month_index = month as i64 + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_index + 2) / 5 + day as i64 - 1;
    let doe = yoe as i64 * 365 + yoe as i64 / 4 - yoe as i64 / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}
