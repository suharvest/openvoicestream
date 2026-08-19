# OpenVoiceStream Agent runtime image (ovs_agent + its apps, incl. voice_arm).
#
# The agent is self-contained: wire-protocol constants are vendored in
# ovs_agent/protocol.py, so it has NO dependency on the SLV server source tree.
#
# Build context = repo root (so `COPY agent` resolves):
#   docker build -f agent/Dockerfile -t ovs-agent:latest .
#
FROM python:3.11-slim

# Build-time mirrors. Default = Tsinghua (CN-fast; the edge build hosts have no
# cross-wall route to deb.debian.org / pypi.org). Override for non-CN builders:
#   --build-arg APT_MIRROR=deb.debian.org --build-arg PIP_INDEX=https://pypi.org/simple
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Point apt at the CN mirror (bookworm ships deb822 sources.list.d/debian.sources;
# older layout uses /etc/apt/sources.list — rewrite whichever exists).
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g; s|security.debian.org|$APT_MIRROR|g" \
            /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
        ca-certificates \
        curl \
        bzip2 \
        build-essential \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slv

# Agent source (self-contained; no SLV server tree needed).
COPY agent /opt/slv/agent

ENV PYTHONPATH=/opt/slv/agent

RUN pip install --no-cache-dir \
        "websockets>=12" \
        "sounddevice>=0.4" \
        "numpy>=1.24" \
        "openai>=1.40" \
        "pyyaml>=6.0" \
        "pysbd>=0.3.4" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "openwakeword>=0.6" \
        "sherpa-onnx==1.13.6" \
        "sentencepiece==0.2.1" \
        "pyaudio>=0.2.13" \
        "onnxruntime>=1.17" \
        "scipy>=1.10" \
    && pip install --no-cache-dir /opt/slv/agent

RUN python -c "import openwakeword.utils; openwakeword.utils.download_models()"

ARG KWS_MODEL_NAME=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
ARG KWS_MODEL_URL=https://ghproxy.net/https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
ARG KWS_MODEL_SHA256=68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
RUN mkdir -p /opt/ovs/models \
    && curl -fL --retry 5 --retry-all-errors "${KWS_MODEL_URL}" -o /tmp/kws.tar.bz2 \
    && echo "${KWS_MODEL_SHA256}  /tmp/kws.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/kws.tar.bz2 -C /opt/ovs/models \
    && test -f "/opt/ovs/models/${KWS_MODEL_NAME}/tokens.txt" \
    && test -f "/opt/ovs/models/${KWS_MODEL_NAME}/en.phone" \
    && rm -f /tmp/kws.tar.bz2

# The generic Agent image owns the cloud/client LLM loop.  The RK1828/speech
# server-loop variant is an explicit deployment choice and must not be enabled
# by the base image (otherwise inject_user_text is rejected by the speech
# service with 409 server-loop_active).
ENV MIC_INDEX=reSpeaker \
    WAKEWORD_MODEL="hey jarvis" \
    WAKEWORD_MODEL_DIR=/opt/ovs/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \
    OVS_AGENT_SERVER_LOOP=0

CMD ["ovs-agent", "run", "voice_arm"]
