//! Ten core devices route, collect pairs behind the opt-in, compute one
//! local update each, and the deltas go through Krum with `f = 3`. A
//! device that did not opt in has no pairs and no upload. Too few uploads
//! fail the round and the active scorer stays. This is code, not a cohort.

use ai_smart_maps_core::confidence::{Environment, RouteNovelty};
use ai_smart_maps_core::contracts::{ModelVersionRecord, TileRecord};
use ai_smart_maps_core::device::{Device, Trip};
use ai_smart_maps_core::graph::{Constraint, Edge, Graph, Node, Source};
use ai_smart_maps_core::reports::Kind;
use ai_smart_maps_core::snap::Destination;
use ai_smart_maps_core::tiles::Origin;
use ai_smart_maps_core::training::TrainingPair;
use ai_smart_maps_later::aggregation::{ActiveScorer, Method, Round, RoundError, Upload};
use ai_smart_maps_later::local_update::{local_update, Pair, Weights, PARAMETERS};

const SUM: &[u8] = include_bytes!("../fixtures/sum_scorer.onnx");

fn graph(long_weight: f64) -> Graph {
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
            edge("long", "a", "c", long_weight, None),
            edge("via", "a", "b", 1.0, Some(1.0)),
            edge("rest", "b", "c", 1.0, None),
        ],
    }
}

fn device(index: u8, opted_in: bool) -> Device {
    let mut device = Device::open(&[index; 32], &[0u8; 32]).unwrap();
    device.install_scorer(
        &ModelVersionRecord {
            id: "scorer-1".to_string(),
            artifact: "scorer-1.onnx".to_string(),
            created_at: "2026-10-01T00:00:00Z".to_string(),
            previous_id: None,
        },
        SUM,
    );
    device
        .load_tile(
            TileRecord {
                id: "tile-1".to_string(),
                observed_at: "2026-10-01T00:00:00Z".to_string(),
                graph: graph(3.0 + f64::from(index) * 0.5),
                signature: None,
            },
            Origin::Local,
        )
        .unwrap();
    if opted_in {
        device.training().opt_in("scorer-1");
    }
    device
        .route(
            "a",
            Destination::NodeId("c"),
            &Trip {
                now: "2026-10-05T00:00:00Z",
                map_age_days: 1.0,
                report_count: 1,
                route_novelty: RouteNovelty::Known,
                environment: Environment::Simple,
            },
        )
        .unwrap();
    device
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

/// What a device would send: a delta only when opted in and pairs exist.
fn upload(device: &mut Device, weights: &Weights) -> Option<Upload> {
    let training = device.training();
    let pairs: Vec<Pair> = training.pairs().iter().map(to_pair).collect();
    if pairs.is_empty() {
        return None;
    }
    Some(Upload {
        opted_in: training.opted_in(),
        delta: local_update(weights, &pairs).ok()?,
    })
}

fn active() -> ActiveScorer {
    ActiveScorer {
        id: "scorer-1".to_string(),
        weights: Weights::seeded(0).0,
    }
}

#[test]
fn ten_opted_in_devices_feed_a_krum_round_with_f_three() {
    let active = active();
    let weights = Weights(active.weights.clone());
    let mut round = Round::new(3);
    let mut deltas = Vec::new();
    for index in 0..10u8 {
        let mut device = device(index, true);
        let upload = upload(&mut device, &weights).expect("an opted-in device has a pair");
        assert_eq!(upload.delta.len(), PARAMETERS);
        deltas.push(upload.delta.clone());
        round.accept(upload).unwrap();
    }
    assert_eq!(round.len(), 10);
    assert!(
        deltas.windows(2).any(|w| w[0] != w[1]),
        "different graphs give different deltas"
    );

    let candidate = round
        .aggregate(Method::Krum { m: 3 }, &active, "scorer-2")
        .unwrap();
    assert_eq!(candidate.previous_id, "scorer-1");
    assert_eq!(candidate.weights.len(), PARAMETERS);
    assert_ne!(candidate.weights, active.weights);
    assert_eq!(
        active.weights,
        Weights::seeded(0).0,
        "the active scorer is unchanged"
    );
}

#[test]
fn a_device_that_did_not_opt_in_sends_nothing() {
    let mut device = device(0, false);
    assert!(device.training().pairs().is_empty());
    assert!(upload(&mut device, &Weights::seeded(0)).is_none());

    let mut round = Round::new(3);
    assert_eq!(
        round.accept(Upload {
            opted_in: false,
            delta: vec![0.0; PARAMETERS],
        }),
        Err(RoundError::NotOptedIn)
    );
    assert!(round.is_empty());
}

#[test]
fn too_few_device_uploads_fail_the_round_and_the_active_scorer_stays() {
    let active = active();
    let weights = Weights(active.weights.clone());
    let mut round = Round::new(3);
    for index in 0..8u8 {
        let mut device = device(index, true);
        round
            .accept(upload(&mut device, &weights).unwrap())
            .unwrap();
    }
    assert_eq!(
        round.aggregate(Method::Krum { m: 3 }, &active, "scorer-2"),
        Err(RoundError::TooFew),
        "eight is below 2f + 3 = 9"
    );
    assert_eq!(active.weights, Weights::seeded(0).0);
}

#[test]
fn a_restored_opted_in_device_still_produces_a_delta() {
    let mut device = device(1, true);
    let weights = Weights::seeded(0);
    let first = upload(&mut device, &weights).expect("opted in before save");
    let blob = device.export_all().unwrap();

    let mut restored = Device::open(&[1u8; 32], &[0u8; 32]).unwrap();
    restored.import_all(&blob).unwrap();
    assert!(restored.training().opted_in());
    assert_eq!(
        restored.training().pairs().len(),
        device.training().pairs().len()
    );
    let again = upload(&mut restored, &weights).expect("opt-in and pairs travel in the blob");
    assert_eq!(again.delta, first.delta);
}

#[test]
fn a_restored_sealed_report_debug_form_has_no_hex() {
    let mut device = device(1, true);
    let hex = device.reporter_key().unwrap().hex();
    device
        .report(
            "long",
            Kind::Hazard,
            Some("flood"),
            &Trip {
                now: "2026-10-05T00:00:00Z",
                map_age_days: 1.0,
                report_count: 1,
                route_novelty: RouteNovelty::Known,
                environment: Environment::Simple,
            },
        )
        .unwrap();
    let blob = device.export_all().unwrap();

    let mut restored = Device::open(&[1u8; 32], &[0u8; 32]).unwrap();
    restored.import_all(&blob).unwrap();
    let sealed = restored.sealed_reports();
    assert_eq!(sealed.len(), 1);
    let shown = format!("{:?}", sealed[0]);
    assert!(shown.contains("redacted"));
    assert!(!shown.contains(&hex));
}
