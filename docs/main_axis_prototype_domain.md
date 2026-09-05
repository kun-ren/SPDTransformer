# Main-baseline axis metrics, prototypes, and domain adversarial training

Branch: `codex/main-axis-prototype-domain`, based on `main` at `202d37c`.
The original main worktree and its uncommitted configuration are not modified.

## Scope

- Q, K, and V projections remain distinct parameters in every attention head.
- Time, frequency, and region always have separate learnable metrics. Heads
  within the same axis use that axis's metric.
- `model.share_metric_across_layers: [false]` keeps metrics separate between
  encoder layers. `true` ties same-axis metrics across compatible layers.
  Compatibility requires the same attention dimension, metric mode, score,
  and rank. Different dimensions form separate sharing groups. A one-layer
  model is unaffected by this switch.
- Metric sharing uses the same Parameter object, not copied initial values.
  The optimizer updates each shared parameter once; gradients accumulate from
  its uses. Loading a checkpoint preserves the constructor's sharing pattern.
- Main's FFN, pooling, preprocessing, and dataset splits are retained. No
  dimension-preserving tangent mixer or mean-anchor gated pooling was ported.

## Class-conditional prototype loss

With `classifier_type: ["mdm"]`, the existing classifier's trainable class
prototypes are reused. Let `Z_i` be a pooled log-SPD feature and `P_c` the
symmetric log-space prototype for class `c`:

```text
L_intra = mean_i ||Z_i - P_yi||_F^2
L_inter = mean_(c<d) max(0, margin - ||P_c - P_d||_F)^2
L_task = cross_entropy + lambda1 * L_intra + lambda2 * L_inter
```

The true class label selects the intra-class prototype. The inter-class term
uses every distinct prototype pair, even if a batch lacks a class. The
prototype-pair norm uses the model's epsilon floor for numerical stability.
No held-out data is used to fit or update prototypes during evaluation.

Configure these keys separately under `training`, or `pretrain` and `fine_tune`:

```yaml
prototype_intra_weight: [0.001]  # lambda1
prototype_inter_weight: [0.01]   # lambda2
prototype_margin: [1.0]
```

These are starting values, not tuned performance claims. Both weights default
to zero when absent. Both zero skips prototype regularization. With the domain
branch disabled and `condition_regularization_weight: [0.0]`, the objective is
exactly CE. Main's existing condition regularization remains available; the
new example configs keep it at zero. Nonzero prototype weights require MDM.

## Domain adversarial branch

```yaml
model:
  domain_adversarial: [true]  # false removes the branch
  domain_hidden_dim: [32]
  domain_dropout: [0.3]
training:  # use pretrain for the pretrain/fine-tune runner
  domain_adversarial_max_weight: [0.03]
  domain_adversarial_warmup_epochs: [10]
  domain_adversarial_schedule: ["dann"]  # also linear or constant
  domain_loss_normalize: [true]
```

The branch consumes the same pooled feature as the task head; the encoder runs
once. Symmetric vectorization with sqrt(2)-scaled off-diagonal entries is
followed by gradient reversal and a small subject classifier. MDM and pooling
task heads support it. At least two training subjects are required.

Only subjects represented in the training indices receive domain IDs.
Unseen subjects receive -1 and never enter domain CE. Domain maps are saved
with checkpoints. Validation/test use only the task path.

For domain CE `L_d`, the encoder receives an additional `-lambda_d * grad(L_d)`
while the domain classifier minimizes positive CE. Optional normalization
divides domain CE by `max(log(number_of_domains), 1)`; GRL compensates for this
so the configured weight remains the effective encoder coefficient against
raw domain CE. During warmup the domain head learns, but its encoder gradient
is zero. A zero maximum weight removes encoder adversarial pressure, not the
domain head; use `domain_adversarial: [false]` to disable it completely.

During single-target fine-tuning, the domain head is frozen and domain loss is
skipped. Task prototypes may still be optimized using fine-tune labels. Each
fine-tune run starts from the corresponding pretrained state, not a preceding
fine-tuned model.

## Configurations and baseline protocols

Run from this worktree using the project's Python environment:

```bash
python src/training/train.py --config configs/train_axis_prototype_domain.yaml
python src/training/train_pretrain_finetune_loro.py --config configs/train_physionet_pretrain_finetune_axis_prototype_domain.yaml
```

The data/cache paths remain relative to the project root. Configure `root_dir`
to an existing data directory if the new worktree has no local data.

The original YAML files are unchanged. Apart from the new feature keys and
output directory, the new configs match their committed main counterparts:

- The grid runner retains both trial-overlap and subject-disjoint settings
  (`allow_subject_overlap: [true, false]`) and five folds. For cross-subject-only
  runs choose `[false]`. It retains main's test-based scheduler/model selection;
  those test scores are therefore not an untouched final generalization test.
- The pretrain/fine-tune runner excludes each target subject from pretraining,
  splits source trials into 70/15/15 train/validation/test, and adapts to that
  target using other runs while splitting the held-out run into validation/test.
  It does not use the experiment branch's global-fold/trial-mixed protocol.

History CSVs report CE, intra/inter prototype penalties, domain CE, domain
accuracy, and GRL weight. Training loss includes domain CE when enabled, while
validation/test loss does not; compare task CE/accuracy rather than those total
losses directly.

## Verification

The added tests check metric/QKV parameter identities, compatible-dimension
sharing, checkpoint round trips, prototype formulas, exact zero-weight CE
gradients, GRL direction, and joint backward passes. Synthetic protocol tests
exercise real optimizers, checkpoint saving, source-only domain mapping, and
fine-tune resets with the domain head frozen. They do not estimate PhysioNet
accuracy or replace a full real-data training run.
