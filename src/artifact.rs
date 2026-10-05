//! Deterministic ONNX bytes for the A1-v5 device shape from a flat
//! weight vector. The graph is the exported 32×2 artifact with its
//! six initializers replaced. The bytes are not a route and carry no
//! reporter key.

use crate::local_update::{split, UpdateError, Weights, PARAMETERS};

const TEMPLATE: &[u8] = include_bytes!("../fixtures/a1v5_device_32x2.onnx");

/// ONNX bytes that [`ai_smart_maps_core::scorer::Host`] can install.
/// The same weights always give the same bytes.
pub fn encode_device_scorer(weights: &Weights) -> Result<Vec<u8>, UpdateError> {
    let layers = split(&weights.0).ok_or(UpdateError::WrongShape)?;
    let replacements = [
        ("net.0.weight", layers.w1),
        ("net.0.bias", layers.b1),
        ("net.2.weight", layers.w2),
        ("net.2.bias", layers.b2),
        ("net.4.weight", layers.w3),
        ("net.4.bias", core::slice::from_ref(&layers.b3)),
    ];
    let mut model = parse(TEMPLATE).map_err(|_| UpdateError::WrongShape)?;
    let graph = model
        .iter_mut()
        .find(|(field, _)| *field == 7)
        .and_then(|(_, wire)| match wire {
            Wire::Len(bytes) => Some(bytes),
            _ => None,
        })
        .ok_or(UpdateError::WrongShape)?;
    let mut nodes = parse(graph).map_err(|_| UpdateError::WrongShape)?;
    let mut replaced = 0;
    for (field, wire) in &mut nodes {
        if *field != 5 {
            continue;
        }
        let Wire::Len(tensor) = wire else {
            continue;
        };
        let mut fields = parse(tensor).map_err(|_| UpdateError::WrongShape)?;
        let name = fields
            .iter()
            .find(|(f, _)| *f == 8)
            .and_then(|(_, w)| match w {
                Wire::Len(bytes) => std::str::from_utf8(bytes).ok(),
                _ => None,
            })
            .ok_or(UpdateError::WrongShape)?
            .to_string();
        let Some((_, values)) = replacements.iter().find(|(wanted, _)| *wanted == name) else {
            continue;
        };
        let raw = floats(values);
        let Some((_, raw_field)) = fields.iter_mut().find(|(f, _)| *f == 9) else {
            return Err(UpdateError::WrongShape);
        };
        *raw_field = Wire::Len(raw);
        *tensor = encode(&fields);
        replaced += 1;
    }
    if replaced != 6 {
        return Err(UpdateError::WrongShape);
    }
    *graph = encode(&nodes);
    debug_assert_eq!(
        layers.w1.len() + layers.b1.len() + layers.w2.len() + layers.b2.len() + layers.w3.len() + 1,
        PARAMETERS
    );
    Ok(encode(&model))
}

#[derive(Clone)]
enum Wire {
    Varint(u64),
    SixtyFour([u8; 8]),
    Len(Vec<u8>),
    ThirtyTwo([u8; 4]),
}

fn parse(bytes: &[u8]) -> Result<Vec<(u32, Wire)>, ()> {
    let mut rest = bytes;
    let mut fields = Vec::new();
    while !rest.is_empty() {
        let (key, after) = take_varint(rest)?;
        rest = after;
        let field = (key >> 3) as u32;
        let wire = match key & 7 {
            0 => {
                let (value, after) = take_varint(rest)?;
                rest = after;
                Wire::Varint(value)
            }
            1 => {
                let (chunk, after) = rest.split_at_checked(8).ok_or(())?;
                rest = after;
                Wire::SixtyFour(chunk.try_into().map_err(|_| ())?)
            }
            2 => {
                let (len, after) = take_varint(rest)?;
                let (payload, after) = after.split_at_checked(len as usize).ok_or(())?;
                rest = after;
                Wire::Len(payload.to_vec())
            }
            5 => {
                let (chunk, after) = rest.split_at_checked(4).ok_or(())?;
                rest = after;
                Wire::ThirtyTwo(chunk.try_into().map_err(|_| ())?)
            }
            _ => return Err(()),
        };
        fields.push((field, wire));
    }
    Ok(fields)
}

fn encode(fields: &[(u32, Wire)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (field, wire) in fields {
        match wire {
            Wire::Varint(value) => {
                put_varint(&mut out, u64::from(*field << 3));
                put_varint(&mut out, *value);
            }
            Wire::SixtyFour(bytes) => {
                put_varint(&mut out, u64::from((*field << 3) | 1));
                out.extend(bytes);
            }
            Wire::Len(bytes) => {
                put_varint(&mut out, u64::from((*field << 3) | 2));
                put_varint(&mut out, bytes.len() as u64);
                out.extend(bytes);
            }
            Wire::ThirtyTwo(bytes) => {
                put_varint(&mut out, u64::from((*field << 3) | 5));
                out.extend(bytes);
            }
        }
    }
    out
}

fn floats(values: &[f32]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for value in values {
        raw.extend(value.to_le_bytes());
    }
    raw
}

fn take_varint(bytes: &[u8]) -> Result<(u64, &[u8]), ()> {
    let mut value = 0u64;
    let mut shift = 0;
    for (index, byte) in bytes.iter().enumerate() {
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, &bytes[index + 1..]));
        }
        shift += 7;
        if shift > 63 {
            return Err(());
        }
    }
    Err(())
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}
