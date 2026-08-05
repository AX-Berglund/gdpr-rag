# Runs the demo on any container host. Streamlit Community Cloud does not need
# this — it installs requirements.txt directly — but a container keeps the
# deployment portable rather than tied to one provider.
FROM python:3.11-slim

# A non-root user with a writable home: Streamlit and Hugging Face both want
# somewhere to cache, and Spaces run as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Spaces route to 7860.
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
