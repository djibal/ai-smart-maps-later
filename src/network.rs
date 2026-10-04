//! Network contract. Bodies are route and report fields.
//! A result past the hard limit is not applied.

use std::collections::HashMap;
use std::time::Duration;

pub const SYNC_TARGET: Duration = Duration::from_secs(2);
pub const SYNC_MAX: Duration = Duration::from_secs(10);
pub const SYNC_HARD_LIMIT: Duration = Duration::from_secs(60);
pub const HAZARD_FRESHNESS_LIMIT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RouteKey {
    pub origin: String,
    pub destination: String,
    pub tile_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedRoute {
    pub edge_ids: Vec<String>,
    pub model_version_id: String,
    pub confidence_score_millis: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReportBody {
    pub id: String,
    pub observed_at: String,
    pub edge_id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Default)]
pub struct EdgeCache {
    routes: HashMap<RouteKey, CachedRoute>,
}

impl EdgeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: RouteKey, route: Option<CachedRoute>, elapsed: Duration) -> bool {
        if elapsed > SYNC_HARD_LIMIT {
            return false;
        }
        if let Some(route) = route {
            self.routes.insert(key, route);
        }
        true
    }

    pub fn get(&self, key: &RouteKey) -> Option<&CachedRoute> {
        self.routes.get(key)
    }
}

pub fn within_hazard_freshness(elapsed: Duration) -> bool {
    elapsed <= HAZARD_FRESHNESS_LIMIT
}

pub fn reports_newer_than<'a>(
    reports: &'a [ReportBody],
    edge_id: &str,
    observed_after: &str,
) -> Vec<&'a ReportBody> {
    reports
        .iter()
        .filter(|report| report.edge_id == edge_id && report.observed_at.as_str() > observed_after)
        .collect()
}
