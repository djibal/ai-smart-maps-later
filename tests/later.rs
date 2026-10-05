use std::time::Duration;

use ai_smart_maps_later::canary::{
    active_id, assigned_id, decide, in_bucket, within_five_percent, within_three_points,
};
use ai_smart_maps_later::mesh::{Broadcast, Receiver};
use ai_smart_maps_later::network::{
    reports_newer_than, within_hazard_freshness, CachedRoute, EdgeCache, ReportBody, RouteKey,
    HAZARD_FRESHNESS_LIMIT, SYNC_HARD_LIMIT, SYNC_MAX, SYNC_TARGET,
};

#[test]
fn one_percent_of_draws_see_the_candidate() {
    assert!(in_bucket(0));
    assert!(in_bucket(100));
    assert!(!in_bucket(1));
    assert_eq!(assigned_id("previous", "candidate", 0), "candidate");
    assert_eq!(assigned_id("previous", "candidate", 1), "previous");
}

#[test]
fn the_candidate_enters_only_inside_both_bars() {
    assert!(within_three_points(98.5, 99.7));
    assert!(within_five_percent(98.5, 99.7));
    let passed = decide("previous", "candidate", 98.0, 99.0, 99.0);
    assert!(passed.passed);
    assert_eq!(active_id(&passed), "candidate");

    let behind = decide("previous", "candidate", 96.0, 99.9, 99.9);
    assert!(!behind.passed);
    assert_eq!(active_id(&behind), "previous");

    let far = decide("previous", "candidate", 90.0, 99.0, 99.0);
    assert!(!far.passed);
    assert!(!within_five_percent(90.0, 99.0));
}

#[test]
fn a_late_or_repeated_broadcast_is_dropped_and_not_stored() {
    let mut receiver = Receiver::new("2026-10-04T00:00:03Z");
    let fresh = Broadcast {
        kind: "hazard".to_string(),
        edge_id: "e1".to_string(),
        observed_at: "2026-10-04T00:00:01Z".to_string(),
    };
    assert!(receiver.accept(fresh.clone()));
    assert!(!receiver.accept(fresh));
    assert!(!receiver.accept(Broadcast {
        kind: "hazard".to_string(),
        edge_id: "e1".to_string(),
        observed_at: "2026-10-04T00:00:00Z".to_string(),
    }));
    assert_eq!(receiver.reports().len(), 1);
    let source = include_str!("../src/mesh.rs");
    assert!(!source.contains("reporter"));
    assert!(!source.contains("signature"));
}

#[test]
fn the_cache_key_is_three_ids_and_a_slow_sync_is_not_applied() {
    assert_eq!(SYNC_TARGET, Duration::from_secs(2));
    assert_eq!(SYNC_MAX, Duration::from_secs(10));
    assert_eq!(SYNC_HARD_LIMIT, Duration::from_secs(60));
    assert_eq!(HAZARD_FRESHNESS_LIMIT, Duration::from_secs(30));
    assert!(within_hazard_freshness(Duration::from_secs(30)));
    assert!(!within_hazard_freshness(Duration::from_secs(31)));

    let mut cache = EdgeCache::new();
    let key = RouteKey {
        origin: "a".to_string(),
        destination: "b".to_string(),
        tile_id: "t".to_string(),
    };
    assert!(!cache.insert(key.clone(), None, Duration::from_secs(61)));
    assert!(cache.get(&key).is_none());
    assert!(cache.insert(
        key.clone(),
        Some(CachedRoute {
            edge_ids: vec!["ab".to_string()],
            model_version_id: "scorer-1".to_string(),
            confidence_score_millis: 1000,
        }),
        Duration::from_secs(2),
    ));
    assert_eq!(cache.get(&key).unwrap().edge_ids, vec!["ab".to_string()]);

    let reports = vec![ReportBody {
        id: "r".to_string(),
        observed_at: "2026-10-04T00:00:02Z".to_string(),
        edge_id: "e1".to_string(),
        kind: "hazard".to_string(),
    }];
    assert!(reports_newer_than(&reports, "e1", "2026-10-04T00:00:01Z").len() == 1);
    assert!(reports_newer_than(&reports, "e1", "2026-10-04T00:00:02Z").is_empty());
}

#[test]
fn later_sources_name_no_person() {
    for name in [
        "src/canary.rs",
        "src/local_update.rs",
        "src/mesh.rs",
        "src/network.rs",
        "src/lib.rs",
    ] {
        let source = std::fs::read_to_string(name).unwrap().to_lowercase();
        for word in ["account", "password", "payment", "passkey", "contact"] {
            assert!(!source.contains(word), "{name} contains {word}");
        }
    }
}
