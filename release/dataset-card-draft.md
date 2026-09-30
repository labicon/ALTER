---
license: cc-by-4.0
---

# ALTER simulation data card

Authors: Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr.

The pretraining selection contains 400 demonstrations: 200 place-return and 200
wipe. Adaptation includes 60 selected multi-arm demonstrations and 60 distilled
single-arm rollouts at the largest budget. Nested manifests retain 20+20,
40+40, and 60+60 membership, mode grouping and ordering. FT-multi omits the
single-arm adaptation domain. Counts exclude base-policy pretraining.

Data and provenance retain distinct namespaces. Portable manifests use logical
artifact references; materialization checks checksums and preserves list order.
Do not rerank examples from relocated absolute paths. Original private records
remain retained separately, and exports record original and export hashes.

Use only with the matching model/preprocessing, camera and action conventions
in the release source. Data are trusted Python pickle artifacts, not a safe
format for arbitrary untrusted downloads. Hardware recordings and their trial
evidence are not included. Original ALTER demonstrations, replay and accompanying
original records are released under CC BY 4.0; see LICENSE and NOTICE. Third-party
software, assets and the paper retain their separate terms.

The provenance bundle contains selected evaluation/selection evidence, not new
experimental results. The original base-training contract is absent, and the
FT-mixed selection-panel wording requires author review. No complete paper
retraining or full evaluation rerun is claimed by preparation tests.

See USAGE.md for downloads and the accompanying code release status.

Paper: [Residual Denoising Enables Sample-Efficient Multi-Agent Coordination on Demand](https://arxiv.org/abs/2609.32129).
