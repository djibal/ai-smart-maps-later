use ai_smart_maps_later::aggregation::{
    krum, trimmed_mean, ActiveScorer, Method, Round, RoundError, Upload, PROVEN_COHORT, PROVEN_F,
    PROVEN_M,
};

fn honest(seed: usize) -> Vec<f32> {
    let wobble = seed as f32 * 0.01;
    vec![1.0 + wobble, 1.0 - wobble, 1.0, 1.0 + wobble]
}

fn flipped() -> Vec<f32> {
    vec![-10.0, -10.0, -10.0, -10.0]
}

fn cohort(honest_count: usize, malicious: usize) -> Vec<Vec<f32>> {
    let mut deltas: Vec<Vec<f32>> = (0..honest_count).map(honest).collect();
    deltas.extend((0..malicious).map(|_| flipped()));
    deltas
}

fn near_honest(values: &[f32]) -> bool {
    values.iter().all(|value| (value - 1.0).abs() < 0.1)
}

#[test]
fn the_proven_setting_is_f3_m3_and_ten_devices() {
    assert_eq!(PROVEN_F, 3);
    assert_eq!(PROVEN_M, 3);
    assert_eq!(PROVEN_COHORT, 10);
}

#[test]
fn krum_recovers_from_three_amplified_sign_flips() {
    let result = krum(&cohort(7, 3), 3, 3).unwrap();
    assert!(near_honest(&result), "{result:?}");
}

#[test]
fn a_cohort_or_f_outside_the_proof_is_unmeasured() {
    assert_eq!(krum(&cohort(8, 3), 3, 3), Err(RoundError::Unmeasured));
    assert_eq!(krum(&cohort(7, 3), 2, 3), Err(RoundError::Unmeasured));
    assert_eq!(krum(&cohort(7, 3), 3, 1), Err(RoundError::Unmeasured));
}

#[test]
fn too_few_deltas_do_not_fall_back_to_a_plain_mean() {
    assert_eq!(krum(&cohort(5, 3), 3, 3), Err(RoundError::TooFew));
    let source = include_str!("../src/aggregation.rs").to_lowercase();
    assert!(!source.contains("fedavg"));
}

#[test]
fn trimmed_mean_needs_trim_at_least_f_over_n() {
    assert_eq!(
        trimmed_mean(&cohort(7, 3), 3, 0.2),
        Err(RoundError::TrimTooSmall)
    );
    let result = trimmed_mean(&cohort(7, 3), 3, 0.3).unwrap();
    assert!(near_honest(&result), "{result:?}");
}

#[test]
fn a_round_keeps_only_opted_in_deltas_and_names_the_previous_scorer() {
    let mut round = Round::new(3);
    assert_eq!(
        round.accept(Upload {
            opted_in: false,
            delta: honest(0),
        }),
        Err(RoundError::NotOptedIn)
    );
    for delta in cohort(7, 3) {
        round
            .accept(Upload {
                opted_in: true,
                delta,
            })
            .unwrap();
    }
    assert_eq!(round.len(), 10);
    let active = ActiveScorer {
        id: "scorer-1".to_string(),
        weights: vec![0.0, 0.0, 0.0, 0.0],
    };
    let candidate = round
        .aggregate(Method::Krum { m: 3 }, &active, "scorer-2")
        .unwrap();
    assert_eq!(candidate.previous_id, "scorer-1");
    assert_eq!(candidate.id, "scorer-2");
    assert!(near_honest(&candidate.weights));
    assert_eq!(active.weights, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_failed_round_returns_an_error_and_the_active_scorer_stays() {
    let mut round = Round::new(3);
    for delta in cohort(4, 0) {
        round
            .accept(Upload {
                opted_in: true,
                delta,
            })
            .unwrap();
    }
    let active = ActiveScorer {
        id: "scorer-1".to_string(),
        weights: vec![0.5; 4],
    };
    assert_eq!(
        round.aggregate(Method::Krum { m: 3 }, &active, "scorer-2"),
        Err(RoundError::TooFew)
    );
    let mut wide = Round::new(3);
    for mut delta in cohort(7, 3) {
        delta.push(0.0);
        wide.accept(Upload {
            opted_in: true,
            delta,
        })
        .unwrap();
    }
    assert_eq!(
        wide.aggregate(Method::Krum { m: 3 }, &active, "scorer-2"),
        Err(RoundError::Mismatch)
    );
    assert_eq!(active.weights, vec![0.5; 4]);
}
