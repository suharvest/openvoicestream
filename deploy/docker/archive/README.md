# Archived one-off overlays

Everything here layered a few files onto an already-published image to produce
a hand-tagged experiment build. None of it is referenced by any compose file,
doc, or CI step, and none of it is a supported way to produce an image.

They are kept (rather than deleted) because several published tags on the
registry were made this way, and these files are the only record of how. Git
history has the full story of each.

## Why this pattern is retired

A season of building like this left three kinds of damage, all hit in practice:

1. **Published images nobody can rebuild.** `slv:v090-demo-b3` and the
   `v0.9.0-n1n2-baked-*` line were assembled by hand from overlays whose base
   tags were never produced (`Dockerfile.jetson.edgellm-v090-overlay` still
   says `TODO-v090-prefix-tag-set-at-P5`). When the one machine that built
   them drifts, the image is an orphan.
2. **Fixes trapped in the wrong layer.** The `libonnxruntime.so.1` symlink and
   `libsentencepiece0` existed in `edgellm-moss-nx` but not in the base, so
   every new variant rediscovered the same rc=127 worker crash.
3. **Per-variant images for what is runtime config.** moss vs customvoice vs
   qwen3tts each got an image. With engines provisioned on demand from
   manifests, that variance is `OVS_PROFILE` plus a manifest — one image
   serves all of them.

## The rules that replaced it

* Every pushed tag is reproducible from a committed Dockerfile plus pinned,
  published inputs (PyPI, HF artifact manifests). If the build needs a file
  that only exists on your machine, it does not get pushed.
* Runtime variance (TTS backend, language, profile) never creates a new
  image. It is a profile and a manifest.
* Shared runtime requirements live in the platform image, not in overlays.
* Experiment overlays stay local. If one graduates, its content moves into
  the platform image or an app image with a real Dockerfile.

See `docs/BUILD_IMAGES.md` for the supported set.
