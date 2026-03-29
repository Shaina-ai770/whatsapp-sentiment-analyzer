# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

CMD ["streamlit", "run", "streamlit_app_enhanced.py", "--server.port=7860", "--server.address=0.0.0.0"]
