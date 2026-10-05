//! The canary decides, the core routes. A device in the 1 percent bucket
//! sees the A1-v5 candidate only when both bars hold; otherwise it keeps
//! `previous_id`. The means below exercise the gate arithmetic. They are
//! not a cohort measurement.

use ai_smart_maps_core::confidence::{Environment, RouteNovelty};
use ai_smart_maps_core::contracts::{ModelVersionRecord, TileRecord};
use ai_smart_maps_core::device::{Device, Trip};
use ai_smart_maps_core::graph::{Constraint, Edge, Graph, Node, Source};
use ai_smart_maps_core::snap::Destination;
use ai_smart_maps_core::tiles::Origin;
use ai_smart_maps_later::canary::{active_id, assigned_id, decide, in_bucket};

const PREVIOUS: &[u8] = include_bytes!("../fixtures/sum_scorer.onnx");
const CANDIDATE: &[u8] = include_bytes!("../fixtures/a1v5_device_32x2.onnx");
const PREVIOUS_ID: &str = "sum-fixture";
const CANDIDATE_ID: &str = "a1v5-32x2";

/// Agreement of the exported A1-v5 artifact on its 1000-sample test, from
/// `fixtures/a1v5_device_32x2.json`. The previous and cloud means below are
/// chosen to sit inside or outside the bars.
const CANDIDATE_MEAN: f64 = 98.9;

fn record(id: &str, previous_id: Option<&str>) -> ModelVersionRecord {
    ModelVersionRecord {
        id: id.to_string(),
        artifact: format!("{id}.onnx"),
        created_at: "2026-10-05T00:00:00Z".to_string(),
        previous_id: previous_id.map(str::to_string),
    }
}

fn graph() -> Graph {
    let node = |id: &str, lat: f64| Node {
        id: id.to_string(),
        lat,
        lon: 0.0,
    };
    let edge = |id: &str, src: &str, dst: &str, weight: f64, hazard: Option<f64>| Edge {
        id: id.to_string(),
        src: src.to_string(),
        dst: dst.to_string(),
        weight,
        constraint: Constraint::Open,
        source: Source::Osm,
        hazard,
        valid_from: None,
        valid_to: None,
    };
    Graph {
        nodes: vec![node("a", 0.0), node("b", 1.0), node("c", 2.0)],
        edges: vec![
            edge("long", "a", "c", 4.0, None),
            edge("via", "a", "b", 1.0, Some(1.0)),
            edge("rest", "b", "c", 1.0, None),
        ],
    }
}

fn device_on_previous() -> Device {
    let mut device = Device::open(&[7u8; 32], &[0u8; 32]).unwrap();
    device.install_scorer(&record(PREVIOUS_ID, None), PREVIOUS);
    device
        .load_tile(
            TileRecord {
                id: "tile-1".to_string(),
                observed_at: "2026-10-01T00:00:00Z".to_string(),
                graph: graph(),
                signature: None,
            },
            Origin::Local,
        )
        .unwrap();
    device
}

fn trip() -> Trip<'static> {
    Trip {
        now: "2026-10-05T00:00:00Z",
        map_age_days: 1.0,
        report_count: 1,
        route_novelty: RouteNovelty::Known,
        environment: Environment::Simple,
    }
}

/// Applies the canary record to the device: the candidate is installed
/// only when it passed. The core never sees the means.
fn roll_out(
    device: &mut Device,
    draw: u32,
    candidate_mean: f64,
    previous_mean: f64,
    cloud_mean: f64,
) -> String {
    if assigned_id(PREVIOUS_ID, CANDIDATE_ID, draw) != CANDIDATE_ID {
        return PREVIOUS_ID.to_string();
    }
    let decision = decide(
        PREVIOUS_ID,
        CANDIDATE_ID,
        candidate_mean,
        previous_mean,
        cloud_mean,
    );
    if decision.passed {
        device.install_scorer(&record(CANDIDATE_ID, Some(PREVIOUS_ID)), CANDIDATE);
    }
    active_id(&decision).to_string()
}

#[test]
fn the_candidate_artifact_is_the_exported_a1_v5_model() {
    assert_eq!(CANDIDATE.len(), 5964);
    assert!(
        include_str!("../fixtures/a1v5_device_32x2.json").contains("\"agreement_pct_1000\": 98.9")
    );
}

#[test]
fn a_device_in_the_bucket_routes_on_the_candidate_when_both_bars_hold() {
    let mut device = device_on_previous();
    assert!(in_bucket(100));
    let active = roll_out(&mut device, 100, CANDIDATE_MEAN, 98.9, 100.0);
    assert_eq!(active, CANDIDATE_ID);

    let outcome = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(outcome.route.model_version_id, CANDIDATE_ID);
    assert_eq!(outcome.route.edge_ids, vec!["long".to_string()]);
    assert!(!outcome.rolled_back);
    let prediction = outcome.prediction.unwrap();
    assert_eq!(prediction.version_id, CANDIDATE_ID);
    assert!(prediction.value.is_finite());
    assert!(device.budget_used() >= (PREVIOUS.len() + CANDIDATE.len()) as u64);
}

#[test]
fn a_device_outside_the_bucket_never_sees_the_candidate() {
    let mut device = device_on_previous();
    let before = device.budget_used();
    assert!(!in_bucket(101));
    let active = roll_out(&mut device, 101, CANDIDATE_MEAN, 98.9, 100.0);
    assert_eq!(active, PREVIOUS_ID);

    let outcome = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(outcome.route.model_version_id, PREVIOUS_ID);
    assert_eq!(outcome.prediction.unwrap().version_id, PREVIOUS_ID);
    assert_eq!(device.budget_used(), before, "nothing was installed");
}

#[test]
fn the_three_point_bar_keeps_the_device_on_previous() {
    let mut device = device_on_previous();
    let active = roll_out(&mut device, 200, 95.0, 98.9, 100.0);
    assert_eq!(active, PREVIOUS_ID, "95.0 is below 98.9 - 3.0");

    let outcome = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(outcome.route.model_version_id, PREVIOUS_ID);
    assert_eq!(outcome.prediction.unwrap().version_id, PREVIOUS_ID);
}

#[test]
fn the_five_percent_bar_keeps_the_device_on_previous() {
    let mut device = device_on_previous();
    let active = roll_out(&mut device, 300, 94.0, 94.0, 100.0);
    assert_eq!(active, PREVIOUS_ID, "94.0 is more than 5 percent off 100.0");

    let outcome = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(outcome.route.model_version_id, PREVIOUS_ID);

    let mut at_the_edge = device_on_previous();
    let active = roll_out(&mut at_the_edge, 300, 95.0, 95.0, 100.0);
    assert_eq!(
        active, CANDIDATE_ID,
        "95.0 is exactly 5 percent off and inclusive"
    );
}
