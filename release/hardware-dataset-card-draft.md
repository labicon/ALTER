# ALTER hardware data — staging draft

Two independently selectable data bundles are prepared:

| Bundle | Files | Bytes | Contents |
| --- | ---: | ---: | --- |
| hardware-demonstrations-data | 501 | 48,137,006,286 | Selected base demonstrations, paired demonstrations, and distilled replay |
| hardware-training-data | 412 | 20,798,158,144 | Original `images.npy` / `trajectory.npz` for 206 prepared records |

The model/metadata bundle provides the portable manifests and catalogs. There are
332 base-training and 32 base-validation records; the high-data adaptation set
contains 69 paired demonstrations and 68 replay records. The low-data set uses
30 pairs and 30 replay records from the same cache pool. Shared data are stored
once. Cardboard replay recordings were policy-generated, not human demonstrations.

High-data sampling has 93,164 two-arm anchors and 44,224 replay anchors; low-data
sampling has 37,319 and 19,921. Original timestamps, indices, phase labels, weights,
action values, image bytes, normalization metadata, grouping and ordering are
preserved. Cartesian xyz uses millimeters; rotations use radians; action rows are
`[x, y, z, roll, pitch, yaw, gripper]`.

These are recovered prepared inputs, not a complete raw hardware backup. Some
original raw recordings and scored physical-trial evidence are absent. Some
prepared records use retained-row ordinal time rather than physical timestamps;
their original per-record provenance documents this. The current-frame training
protocol is preserved rather than reinterpreted as a temporal dataset.

Original data/cache bytes are unchanged; export receipts and checksum manifests
record their identities. Path-bearing metadata is sanitized in independent copies.
Representative inspection of 59 frames showed robot workspaces and task objects,
with no obvious faces or private text in those samples. This was not an exhaustive
review of every frame and does not establish release permission for recordings.

Status: local staging only. Hardware data release terms and publication approval
remain to be finalized. Existing public simulation data are unchanged.
