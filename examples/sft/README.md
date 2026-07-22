# SFT optimization and experiment report

This document records the follow-up optimization pass over the SFT domain
introduced by [PR #209](https://github.com/Tencent-Hunyuan/UniRL/pull/209).
The goal was not to add a second training stack, but to remove avoidable work
around the existing shared `TrainStack` while preserving its loss and
checkpoint semantics.

The performance figures below were measured against upstream `main` at
`2e27f423cbc560198479e4656e9a1a0caeef6997`. The implementation has since
been rebased onto current `main`; the recorded timings remain historical A/B
evidence rather than new measurements on the rebased tree.

## Decision criteria

Each candidate was isolated from the others. Performance runs used three
baseline and three variant arms in `baseline, variant, variant, baseline,
baseline, variant` order, with fixed data/seed and warmup steps excluded.

A performance change was retained only when:

- token/mask/latent outputs remained equivalent;
- loss behavior remained consistent with the baseline;
- the target phase or steady-state step median improved by at least 5%; and
- the optimization had a fail-safe path when its assumptions were not true.

Correctness changes were retained when they made invalid state fail loudly
without breaking a valid resume or an explicitly configured evaluation split.

## Data and evaluation correctness

### Manifest-aware resume

`SupervisedDataSource.state_dict()` now stores:

- the exact epoch and position cursor;
- the shuffle seed and enabled/disabled mode;
- a streaming SHA-256 fingerprint of the train manifest; and
- the number of manifest records.

On resume, a changed fingerprint, record count, shuffle seed, or shuffle mode
raises before training. Old sidecars without a fingerprint or shuffle mode
remain readable with a warning only for the missing fingerprint.

The A/B probe replaced a manifest with different content but the same number
of rows. The old implementation resumed silently; the new implementation
rejected it. An unchanged manifest resumed to the exact same subsequent batch
sequence.

The fingerprint covers the manifest bytes. Media files referenced by the
manifest are versioned separately by the encoder-cache key.

### Explicit evaluation data

SFT recipes no longer default `eval_manifest_path` to the train manifest.
When `SFT_EVAL_DATA` is unset, validation is disabled and the trainer skips
eval calls. Users can still explicitly point `SFT_EVAL_DATA` at the train
manifest when that is intentional.

The A/B probe changed the implicit eval row count from six train rows to zero,
while an explicitly configured eval manifest continued to produce all six
rows.

## Worker-side preprocessing

### Batched response tokenization

`ARSupervisedTrackBuilder` now calls the tokenizer once for the response batch,
then applies the existing per-sample truncation, EOS, and eval-padding mask
rules.

With the Qwen3 tokenizer and 64 responses, the isolated tokenizer phase
improved by `3.41x`. Qwen3 end-to-end step time was unchanged because
tokenization was already a small fraction of the step, but the optimized path
is output-equivalent and removes Python call overhead for larger batches.

### Parallel image loading

VLM condition images and diffusion target images are loaded through a bounded
thread pool. Input order and `None` rows are preserved, and files are fully
converted to RGB before their handles are closed.

On unique cold files, the PIL/Ceph phase improved by `3.51x`. In the Qwen-VL
training run, cold first-epoch build time was about `2.33s`; once the OS page
cache was warm, both arms converged to roughly `0.026s`, so repeated-run step
medians were unchanged. The change is retained for first-epoch and cold-storage
workloads.

`image_load_workers=1` restores serial loading. The default is four workers.

### Qwen-VL next-batch CPU prefetch

Qwen-VL recipes enable a one-batch CPU prefetch. The track builder tokenizes
responses and loads images on a background thread while the current GPU train
step runs.

The data source uses a peek/commit protocol:

1. `peek_samples()` materializes the next batch without advancing the
   checkpointed cursor.
2. The worker begins CPU preparation for that batch.
3. After the current optimizer step, `commit_peeked_samples()` advances the
   cursor when the prefetched batch becomes current.

This keeps sidecars exact even if training stops while a future batch is
already prepared. Eval batches with different records leave the pending train
prefetch intact; an identical eval batch may safely reuse it, after which the
train build recomputes that CPU preparation without changing the data cursor.

On warm Qwen-VL data, build time changed from `0.028s` to `0.019s` (about 32%
faster). The rounded end-to-end step median remained `0.6s` in both arms.

Qwen-VL keeps the pipeline's `pad_to_max_length: false` default; dynamic
per-shard padding is correct for SFT and avoids the known fixed-padding
generation risk.

## Frozen encoder caches

`DiffusionSupervisedTrackBuilder` supports two independent, opt-in disk caches:

- `cache_text_conditions` for frozen text-encoder outputs; and
- `cache_vae_latents` for deterministic frozen-VAE target latents.

The cache is disabled by default. A recipe can expose it with:

```yaml
track_builder:
  cache_dir: ${oc.env:SFT_CACHE_DIR,null}
  cache_fingerprint: ${oc.env:SFT_CACHE_FINGERPRINT,null}
  cache_text_conditions: ${oc.decode:${oc.env:SFT_CACHE_TEXT_CONDITIONS,false}}
  cache_vae_latents: ${oc.decode:${oc.env:SFT_CACHE_VAE_LATENTS,false}}
  cache_max_entries: ${oc.decode:${oc.env:SFT_CACHE_MAX_ENTRIES,4096}}
```

Enabling either cache requires both `cache_dir` and an explicit
`cache_fingerprint`. The fingerprint should identify the model revision,
encoder configuration, and preprocessing contract.

Safety rules:

- encoder parameters are inspected lazily after backend/LoRA setup;
- any trainable, training-mode, or uninspectable encoder disables its cache;
- entries are namespaced by the explicit model/config fingerprint;
- text keys include the prompt and condition parameters;
- VAE keys include media path, inode, size, mtime, ctime, target resolution,
  and resize policy;
- writes are atomic, reads use PyTorch's restricted weights-only loader, and
  each cache kind is periodically evicted toward its configured bound
  (concurrent workers may transiently exceed it); and
- cached values are reassembled in original batch order.

The Bagel recipe exposes only the VAE cache. Its prompt-context prefill is
collective, so asymmetric per-rank text-cache hits could deadlock. Bagel VAE
latents are safe because the VAE is frozen and deterministic.

Measured second-epoch results:

- SD3 text conditions: build `0.1755s -> 0.1400s`, step `0.6s -> 0.5s`
  (`1.20x`);
- SD3 VAE latents: build `0.1765s -> 0.0475s`, step `0.6s -> 0.4s`
  (`1.50x`); and
- Bagel VAE latents: `0.03928s -> 0.00362s` (`10.86x`), with maximum latent
  absolute difference `0`.

## Candidates not merged

### Batched micro loss weights

A prefix-sum implementation removed repeated per-micro `.item()` calls and
reused the token counts for logging. Qwen3 3+3 A/B runs measured `1.5s` in
both arms (`1.00x`), below the retention threshold, so the implementation was
reverted.

### Fused build/train dispatch

The earlier F3 experiment fused track build with train/eval dispatch. Single
GPU order reversal removed the apparent regression, while DP8 measured
`0.4s -> 0.4s`. The source and stack already share a worker and transport
tracks through the zero-copy tensor store, so this API complexity was not
retained.

### Diffusion resize and VAE posterior sampling

SD3 and Bagel intentionally use a fixed target resolution and deterministic
posterior means. An offline non-square characterization showed the expected
trade-off:

- stretch preserves edge content but changes geometry; and
- center crop preserves local geometry but removes edge content.

There was no dataset-independent winner, so the documented stretch behavior
remains unchanged. Posterior sampling remains disabled because it would make a
stored target latent differ from what decode reproduces.

## Validation

The post-rebase review on current `main` passed all repository pre-commit
hooks, composed all nine affected SFT recipes, and reran focused CPU checks for
manifest snapshot integrity, exact resume, prefetch cursor semantics, batched
tokenization, parallel image loading, restricted cache loading, and cache
invalidation/recovery.

The original optimization branch passed:

- 17 CPU unit tests (before `main` removed the repository test tree) covering
  manifest identity, exact resume, explicit eval, batch tokenization, parallel
  image loading, prefetch cursor semantics, cache guards/invalidations, atomic
  bounded cache behavior, and the then-current Bagel optional flash-attention
  fallback;
- Qwen3 and Qwen-VL LoRA SFT smoke runs;
- all SD3 text-condition and VAE-cache A/B arms;
- a real Bagel VAE cache A/B and a one-step full Bagel SFT smoke; and
- a checkpoint sidecar resume that restored the data cursor and reached the
  next optimizer step.

Example launch:

```bash
ENTRY=train_sft \
SFT_DATA=/path/to/train.jsonl \
SFT_EVAL_DATA=/path/to/validation.jsonl \
bash examples/run_experiment_single_node.sh sft/qwen3_sft_lora
```

Example SD3 launch with both frozen caches:

```bash
ENTRY=train_sft \
SFT_DATA=/path/to/train.jsonl \
SFT_EVAL_DATA=/path/to/validation.jsonl \
SFT_CACHE_DIR=/path/to/cache \
SFT_CACHE_FINGERPRINT=sd35-medium-revision-and-preprocess-v1 \
SFT_CACHE_TEXT_CONDITIONS=true \
SFT_CACHE_VAE_LATENTS=true \
bash examples/run_experiment_single_node.sh sft/sd3_sft_lora
```
