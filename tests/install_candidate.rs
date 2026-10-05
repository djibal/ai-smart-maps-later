//! A successful Krum candidate becomes ONNX bytes the core host can
//! install. A failed round does not touch the installed artifact. Code,
//! not a cohort.

use ai_smart_maps_core::confidence::{Environment, RouteNovelty};
use ai_smart_maps_core::contracts::{ModelVersionRecord, TileRecord};
use ai_smart_maps_core::device::{Device, Trip};
use ai_smart_maps_core::graph::{Constraint, Edge, Graph, Node, Source};
use ai_smart_maps_core::snap::Destination;
use ai_smart_maps_core::tiles::Origin;
use ai_smart_maps_core::training::TrainingPair;
use ai_smart_maps_later::aggregation::{ActiveScorer, Method, Round, RoundError, Upload};
use ai_smart_maps_later::artifact::encode_device_scorer;
use ai_smart_maps_later::local_update::{local_update, score, Pair, Weights, PARAMETERS};

const PREVIOUS: &[u8] = include_bytes!("../fixtures/a1v5_device_32x2.onnx");
const PREVIOUS_ID: &str = "a1v5-32x2";
const CANDIDATE_ID: &str = "scorer-2";

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

fn device_on(bytes: &[u8], id: &str, previous_id: Option<&str>) -> Device {
    let mut device = Device::open(&[2u8; 32], &[0u8; 32]).unwrap();
    device.install_scorer(&record(id, previous_id), bytes);
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

fn pair() -> Pair {
    Pair {
        left: [1.0, 4.0, 0.0, 1.0],
        right: [2.0, 2.0, 1.0, 1.0],
        left_has_lower_cost: true,
    }
}

fn uploads(count: usize, weights: &Weights) -> Vec<Upload> {
    (0..count)
        .map(|i| {
            let mut left = pair();
            left.left[1] += i as f32 * 0.1;
            Upload {
                opted_in: true,
                delta: local_update(weights, &[left]).unwrap(),
            }
        })
        .collect()
}

#[test]
fn encoded_weights_score_the_same_on_the_host_as_the_local_forward() {
    let weights = Weights::seeded(0);
    let bytes = encode_device_scorer(&weights).unwrap();
    assert_eq!(encode_device_scorer(&weights).unwrap(), bytes);
    assert_eq!(weights.0.len(), PARAMETERS);

    let mut device = device_on(&bytes, CANDIDATE_ID, None);
    let outcome = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(outcome.route.model_version_id, CANDIDATE_ID);
    let prediction = outcome.prediction.unwrap();
    let expected = score(&weights, &[1.0, 4.0, 0.0, 1.0]).unwrap();
    assert!(
        (prediction.value - expected).abs() < 1e-4,
        "host {} local {expected}",
        prediction.value
    );
}

#[test]
fn a_successful_krum_candidate_installs_as_the_next_version() {
    let active = ActiveScorer {
        id: PREVIOUS_ID.to_string(),
        weights: Weights::seeded(0).0,
    };
    let weights = Weights(active.weights.clone());
    let mut round = Round::new(3);
    for upload in uploads(10, &weights) {
        round.accept(upload).unwrap();
    }
    let candidate = round
        .aggregate(Method::Krum { m: 3 }, &active, CANDIDATE_ID)
        .unwrap();
    let bytes = encode_device_scorer(&Weights(candidate.weights.clone())).unwrap();

    let mut previous = device_on(PREVIOUS, PREVIOUS_ID, None);
    let before = previous
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(before.route.model_version_id, PREVIOUS_ID);

    // A fresh device: the router cache would otherwise keep the first
    // route's model version on the same origin, destination, and tile.
    let mut device = device_on(PREVIOUS, PREVIOUS_ID, None);
    device.install_scorer(&record(CANDIDATE_ID, Some(PREVIOUS_ID)), &bytes);
    let after = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(after.route.model_version_id, CANDIDATE_ID);
    assert_eq!(candidate.previous_id, PREVIOUS_ID);
    assert!(!after.rolled_back);
    let after_score = after.prediction.unwrap().value;
    assert!(after_score.is_finite());
    assert_ne!(after_score, before.prediction.unwrap().value);
}

#[test]
fn a_failed_round_leaves_the_installed_artifact_unchanged() {
    let mut device = device_on(PREVIOUS, PREVIOUS_ID, None);
    let before = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    let installed = before.prediction.unwrap().value;

    let active = ActiveScorer {
        id: PREVIOUS_ID.to_string(),
        weights: Weights::seeded(0).0,
    };
    let weights = Weights(active.weights.clone());
    let mut round = Round::new(3);
    for upload in uploads(8, &weights) {
        round.accept(upload).unwrap();
    }
    assert_eq!(
        round.aggregate(Method::Krum { m: 3 }, &active, CANDIDATE_ID),
        Err(RoundError::TooFew)
    );

    let after = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(after.route.model_version_id, PREVIOUS_ID);
    assert_eq!(after.prediction.unwrap().value, installed);
}

fn to_pair(pair: &TrainingPair) -> Pair {
    let features = |f: &ai_smart_maps_core::scorer::Features| {
        [
            f.edge_count as f32,
            f.weight_sum as f32,
            f.hazard_sum as f32,
            f.hazard_missing as f32,
        ]
    };
    Pair {
        left: features(&pair.left),
        right: features(&pair.right),
        left_has_lower_cost: pair.left_has_lower_cost,
    }
}

#[test]
fn one_device_collects_pairs_updates_encodes_and_installs() {
    let mut device = device_on(PREVIOUS, PREVIOUS_ID, None);
    device.training().opt_in(PREVIOUS_ID);
    device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    let pairs: Vec<Pair> = device.training().pairs().iter().map(to_pair).collect();
    assert_eq!(pairs.len(), 1);

    let start = Weights::seeded(0);
    let delta = local_update(&start, &pairs).unwrap();
    let stepped = Weights(
        start
            .0
            .iter()
            .zip(&delta)
            .map(|(weight, step)| weight + step)
            .collect(),
    );
    let bytes = encode_device_scorer(&stepped).unwrap();

    device.install_scorer(&record(CANDIDATE_ID, Some(PREVIOUS_ID)), &bytes);
    let after = device
        .route("a", Destination::NodeId("c"), &trip())
        .unwrap();
    assert_eq!(after.route.model_version_id, CANDIDATE_ID);
    let host = after.prediction.unwrap().value;
    let expected = score(&stepped, &[1.0, 4.0, 0.0, 1.0]).unwrap();
    assert!(
        (host - expected).abs() < 1e-4,
        "host {host} local {expected}"
    );
}
