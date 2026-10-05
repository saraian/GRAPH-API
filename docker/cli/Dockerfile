# Small controller image; neither ROS nor GPU libraries are needed for CLI help.
FROM debian:bookworm-slim
ARG BUILD_REVISION=manual
LABEL graphapi.cli="1" graphapi.cli.build="${BUILD_REVISION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-yaml python3-pytest docker.io bash git openssh-client rsync \
    curl ca-certificates iproute2 iputils-ping procps \
    && rm -rf /var/lib/apt/lists/*
COPY graphapi_cli /opt/graphapi-cli/graphapi_cli
ENV PYTHONPATH=/opt/graphapi-cli PYTHONUNBUFFERED=1 GRAPHAPI_CLI_CONTAINER=1
ENTRYPOINT ["python3", "-m", "graphapi_cli.cli"]
