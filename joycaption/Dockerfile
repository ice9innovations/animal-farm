FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

ARG JOYCAPTION_REPO_URL=https://github.com/fpgaminer/joycaption.git
RUN git clone --depth 1 "$JOYCAPTION_REPO_URL" /app/joycaption-src

COPY emoji_mappings.json /app/
COPY mwe.txt /app/
COPY joycaption_analyzer.py /app/
COPY REST.py /app/

ENV PORT=7797 \
    PRIVATE=False \
    MODEL_ID=fancyfeast/llama-joycaption-beta-one-hf-llava \
    MODEL_DIR=/models \
    DEVICE=auto \
    PRECISION=8bit \
    AUTO_UPDATE=false

EXPOSE 7797

CMD ["python", "REST.py"]
