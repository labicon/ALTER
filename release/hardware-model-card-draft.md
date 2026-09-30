# ALTER hardware models — staging draft

Seven hardware model/statistics pairs are prepared: the 100,000-step single-arm
base and H69, S69, FT_mixed, H30, S30, and FT-mixed-30 at 25,000 steps. The author
confirmed the six 25k checkpoints were used for the paper's hardware results.
H69/H30 use a frozen base with a coordination head; S69/S30 train a full policy
from scratch; the FT variants initialize the full policy from the base EMA.

The model bundle contains 226 files (819,478,346 bytes), including portable
statistics, high/low training manifests, record metadata, model/data catalogs,
and two saved camera frames for offline inference. All checkpoint bytes remain
unchanged. The accompanying export receipts record original and portable hashes.
The camera frames retain their own dataset release terms when approved.

Path-bearing metadata uses `artifact://` references for included artifacts and
`provenance://sha256/` identifiers for historical references not distributed.
The private original-path mapping is not included. Numerical arrays, architecture,
normalization, splits, ordering, sampling weights, and historical hashes are
preserved. Six checkpoint-selection status fields record the author's confirmation.

Some fine-tuning metadata records a historical base-statistics hash whose exact
file was not recovered. That hash remains unchanged; the selected base/statistics
pair was tested with explicit artifact overrides and is not represented as that
missing historical file. This release supports functional offline checks and is
not a claim of exact full training reproduction.

The prepared public profile uses Python 3.10, PyTorch 2.4.1+cpu, torchvision
0.19.1+cpu, and NumPy 1.26.4. Seven models pass CPU sampling on saved frames.
Short training checks use recovered original caches; they do not rerun physical
trials or estimate new hardware success rates. Physical operation depends on the
private, separately maintained ICON_Arm project, which is not bundled.

Status: local staging only. Hub revisions and hardware model/data licenses must
be finalized before publication. The existing simulation release remains unchanged.
