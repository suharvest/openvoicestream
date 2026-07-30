# Standalone Wyoming adapter. Deliberately NOT part of the voice image: it only
# speaks HTTP/WS to an existing seeed-local-voice instance, so both integration
# shapes (we-are-the-brain, HA-is-the-brain) can run side by side.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    SLV_BASE_URL=http://127.0.0.1:8621 \
    WYOMING_HOST=0.0.0.0 \
    WYOMING_STT_PORT=10300 \
    WYOMING_TTS_PORT=10200

COPY pyproject.toml /app/
COPY wyoming_slv /app/wyoming_slv
RUN pip install --no-cache-dir .

EXPOSE 10300 10200

CMD ["python", "-m", "wyoming_slv"]
