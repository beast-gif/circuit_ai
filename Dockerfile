# The backend image for Render.
#
# The one thing that makes this image different from a plain Python one is ngspice: without it
# every answer comes back "not checked by simulation", which is the feature the project is built
# around. apt has it, so it is one line -- but it has to be a Docker service, not a Python
# service, because Render's Python runtime cannot install system packages.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ngspice \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so a code change does not reinstall them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# config.yaml keeps the Windows path for local runs; this overrides it inside the container
ENV NGSPICE_PATH=/usr/bin/ngspice \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Render provides $PORT. One worker on purpose: a thread_id lives in this process's memory, so a
# second worker would not recognise a session started by the first.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
