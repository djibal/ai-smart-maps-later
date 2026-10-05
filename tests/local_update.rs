use ai_smart_maps_later::local_update::{
    local_update, loss, score, Pair, UpdateError, Weights, HIDDEN, INPUTS, LAYERS, LEARNING_RATE,
    MARGIN, PARAMETERS,
};

fn pairs() -> Vec<Pair> {
    vec![
        Pair {
            left: [2.0, 5.0, 0.2, 0.0],
            right: [4.0, 30.0, 2.5, 1.0],
            left_has_lower_cost: true,
        },
        Pair {
            left: [3.0, 12.0, 1.0, 0.0],
            right: [2.0, 6.0, 0.0, 1.0],
            left_has_lower_cost: false,
        },
        Pair {
            left: [1.0, 4.0, 0.0, 1.0],
            right: [2.0, 2.0, 1.0, 1.0],
            left_has_lower_cost: true,
        },
    ]
}

#[test]
fn the_shape_is_the_a1_v5_device_model() {
    assert_eq!((INPUTS, HIDDEN, LAYERS), (4, 32, 2));
    assert_eq!(PARAMETERS, 1249);
    assert_eq!(MARGIN, 1.0);
    assert_eq!(LEARNING_RATE, 1e-3);
    assert!(Weights::seeded(0).is_well_formed());
    assert_ne!(Weights::seeded(0), Weights::seeded(1));
    assert_eq!(Weights::seeded(7), Weights::seeded(7));
}

#[test]
fn one_step_lowers_the_margin_loss_and_is_deterministic() {
    let weights = Weights::seeded(0);
    let before = loss(&weights, &pairs()).unwrap();
    assert!(before > 0.0);

    let delta = local_update(&weights, &pairs()).unwrap();
    assert_eq!(delta.len(), PARAMETERS);
    assert_eq!(delta, local_update(&weights, &pairs()).unwrap());
    assert!(delta.iter().any(|d| *d != 0.0));

    let stepped = Weights(weights.0.iter().zip(&delta).map(|(w, d)| w + d).collect());
    let after = loss(&stepped, &pairs()).unwrap();
    assert!(after < before, "{after} against {before}");
}

#[test]
fn a_satisfied_pair_contributes_nothing() {
    let mut weights = Weights::seeded(3);
    // Make the first output weight large so the margin is already met for
    // a pair whose preferred side has the bigger feature values.
    let pair = Pair {
        left: [10.0, 10.0, 10.0, 10.0],
        right: [0.0, 0.0, 0.0, 0.0],
        left_has_lower_cost: true,
    };
    for _ in 0..200 {
        let delta = local_update(&weights, &[pair]).unwrap();
        for (w, d) in weights.0.iter_mut().zip(&delta) {
            *w += d;
        }
    }
    let gap = score(&weights, &pair.left).unwrap() - score(&weights, &pair.right).unwrap();
    assert!(gap >= MARGIN, "{gap}");
    assert_eq!(loss(&weights, &[pair]).unwrap(), 0.0);
    assert!(local_update(&weights, &[pair])
        .unwrap()
        .iter()
        .all(|d| *d == 0.0));
}

#[test]
fn a_wrong_shape_or_no_pairs_is_refused() {
    assert_eq!(
        local_update(&Weights(vec![0.0; 10]), &pairs()),
        Err(UpdateError::WrongShape)
    );
    assert_eq!(
        local_update(&Weights::seeded(0), &[]),
        Err(UpdateError::NoPairs)
    );
    assert_eq!(
        score(&Weights(vec![]), &[0.0; 4]),
        Err(UpdateError::WrongShape)
    );
}
