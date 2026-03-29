FROM python:3.11-slim

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app_enhanced.py", "--server.port=7860", "--server.address=0.0.0.0"]
