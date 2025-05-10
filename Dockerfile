# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# # Install ollama
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # # Pull ollama model
# RUN ollama pull llama3.2:3b

# Set environment variables
ENV PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
ENV PYTHONUNBUFFERED=1

# Expose ports for Streamlit and ollama
EXPOSE 8501 

# Start ollama and Streamlit
CMD ["/bin/bash", "-c", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]