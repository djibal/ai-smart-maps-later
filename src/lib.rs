//! Later phases, outside the routing core.
//! The canary, the mesh accept rules, and the network contract live here.
//! User counts, the neighborhood pilot, and the independent audit are not
//! results of this crate.

pub mod aggregation;
pub mod artifact;
pub mod canary;
pub mod local_update;
pub mod mesh;
pub mod network;
