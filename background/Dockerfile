FROM rust:1.75-bookworm

# System packages (Python + Tkinter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    python3-tk tk \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Create a venv inside the IMAGE (not in your repo)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python deps into the venv
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt