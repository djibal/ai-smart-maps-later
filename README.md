An aggregation round keeps weight deltas only from devices that opted in. Krum with `f = 3` and `m = 3` is the aggregator, proven for at most 10 devices. Trimmed mean runs only when `trim` is at least `f / n`. A cohort or an `f` outside that proof is refused. A failed round returns an error and leaves the active scorer unchanged. The candidate names the active scorer as `previous_id`.

The canary assigns about 1 percent of device-local draws to a candidate model version. A candidate more than 5 percent off the cloud reference does not enter. A candidate more than 3 percentage points behind the previous version leaves the device on `previous_id`. The record stores version ids and pass or fail.

A mesh broadcast is a consensus kind, an edge id, and an observation time. The receiver drops a broadcast more than 2 seconds behind its clock, and drops a duplicate. An accepted broadcast becomes a local report. The broadcast itself is not stored.

An edge-cache key is an origin node id, a destination node id, and a tile id. Community sync past 60 seconds is not applied. Hazard freshness is 30 seconds. This crate does not record a user count, a neighborhood pilot, or an independent audit.
