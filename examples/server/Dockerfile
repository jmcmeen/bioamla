# A bioamla server in a container. Build with the extras the server will need:
#
#   docker build -t bioamla-server .                           # the slim default below
#   docker build --build-arg BIOAMLA_EXTRAS=all,cli -t bioamla-server .   # every domain (torch: large)
#
# compose.yaml next to this file runs it with the workspace root and the recordings
# mounted and the token as a secret.
FROM python:3.12-slim

# ffmpeg decodes anything that isn't WAV, FLAC or OGG, and renders spectrogram videos.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Which extras to install: at least server and cli; audio for signal processing;
# all for the ML stack (training and prediction need torch, which is large).
ARG BIOAMLA_EXTRAS=server,cli,audio
RUN pip install --no-cache-dir "bioamla[${BIOAMLA_EXTRAS}]"

# An unprivileged user; the mounts below are owned by whoever runs compose, so match
# its uid with the user: option in compose.yaml if files come out with the wrong owner.
RUN useradd --create-home --uid 1000 bioamla \
    && mkdir -p /workspaces /recordings \
    && chown bioamla /workspaces
USER bioamla
WORKDIR /home/bioamla

EXPOSE 8765
# Inside the container the server must listen on every interface; compose publishes the
# port on the host's loopback only, so it stays localhost-only on the host.
ENTRYPOINT ["bioamla", "serve", "--host", "0.0.0.0", "--port", "8765"]
CMD ["--root", "/workspaces", "--token-file", "/run/secrets/bioamla_token", "--allow-link", "/recordings"]
