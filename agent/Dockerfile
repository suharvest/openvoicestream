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
        "pyaudio>=0.2.13" \
        "onnxruntime>=1.17" \
        "scipy>=1.10" \
    && pip install --no-cache-dir /opt/slv/agent

RUN python -c "import openwakeword.utils; openwakeword.utils.download_models()"

# The generic Agent image owns the cloud/client LLM loop.  The RK1828/speech
# server-loop variant is an explicit deployment choice and must not be enabled
# by the base image (otherwise inject_user_text is rejected by the speech
# service with 409 server-loop_active).
ENV MIC_INDEX=reSpeaker \
    WAKEWORD_MODEL="hey jarvis" \
    OVS_AGENT_SERVER_LOOP=0

CMD ["ovs-agent", "run", "voice_arm"]
