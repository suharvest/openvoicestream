# Provenance for the two on-hardware validations

The numbers in `docs/perf/whisper-cross-device-20260827.md` under "Both untested
paths, closed on hardware" rest on artefacts that are not all committable — a
44 MB TensorRT plan is device- and version-specific and belongs on the device,
not in git. What is committable is the chain that ties the committed files
to it. Two of its five links can be checked from a clone with network access;
two require the device; one is a property of the code. None of them establishes
authorship — see the note under the table.

## The TensorRT engine (Orin Nano, TensorRT 10.3.0)

| link | value | how to check |
|---|---|---|
| ONNX on HuggingFace | `9e8acc7d4d4776d5…` | `curl -s https://huggingface.co/api/models/harvestsu/whisper-edge/tree/main/encoder/jetson` |
| ONNX after provisioning, on device | `9e8acc7d4d4776d5…` | `sha256sum whisper-fresh/encoder/jetson/enc_base_30s.onnx` |
| `spec.onnx_sha256` in the sidecar | `9e8acc7d4d4776d5` | `enc_base_30s_bf16.plan.buildinfo.json` (committed) |
| `plan_sha256` in the sidecar | `e8217fd2504f1e99…` | same file |
| the plan itself, on device | `e8217fd2504f1e99…` | `sha256sum whisper-fresh/encoder/jetson/enc_base_30s_bf16.plan` |

`_build_whisper_trt_engine` writes `{trt, spec, plan_bytes, plan_sha256}`;
`_build_sensevoice_trt_engine`, the only other sidecar writer in the repo,
writes `{trt, spec}` with a different spec shape. So this file is **consistent
with** the Whisper writer and inconsistent with the other one.

That is a weaker statement than it may look, and worth stating precisely: a
matching shape shows the file *could* have been written by that function, not
that it *was* written by the run described here. Nothing in an artefact can
establish its own authorship. What the hashes do establish is narrower and
still useful — that a plan with this exact content was built from an ONNX with
this exact content, and that the ONNX is the published one.

`selfbuilt_en.json` is the transcription produced with that plan. It carries no
engine hash of its own — the runner does not record one, which is a gap worth
closing if this is repeated — so the link from it to the plan is the run itself,
not a field. What the file does support independently: all ten corpus files
present, per-row timings different from every other run in this directory
(`orin-nano-en_30s.json`, the hand-built plan), and `rtf` consistent with its own
`encoder_ms + decoder_ms` over `duration_s`.

## The published wheel (RK3588)

| link | value |
|---|---|
| image `openvoicestream:rk-whisper-rel` | `pip show voxedge` → `0.0.12a0` |
| provisioned into a fresh directory | `whisper_encoder_base_10s.rknn` 42654381 B, `decoder_model.onnx` 159454910 B, `decoder_with_past_model.onnx` 156260217 B |
| those sizes | identical to the published repo's, listed in the perf document |

No JSON was produced for this one: it was a single `/asr` request checking that
the profile boots and transcribes from the released wheel, not a benchmark. The
transcript is quoted in the perf document. **This validation is attested rather
than reproducible from the repository** — the image and the model directory are
on the device.
