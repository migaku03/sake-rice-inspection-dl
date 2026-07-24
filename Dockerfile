FROM python:3.10-slim

WORKDIR /app

# Japanese-capable font for matplotlib: class names (心白, 基白, ...) and
# source filenames used in plot titles/labels are Japanese, and the base
# image's default font (DejaVu Sans) has no CJK glyphs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

ENV PYTHONPATH=/app/src

CMD ["pytest", "-q"]
