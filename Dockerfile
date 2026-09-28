FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY aia_psf_deconvolution ./aia_psf_deconvolution
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e . \
    && useradd --create-home --uid 10001 aia

USER aia

ENTRYPOINT ["python", "-m", "aia_psf_deconvolution"]
CMD ["all"]

