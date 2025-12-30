FROM python:3.10-slim

# Install system dependencies for building llama-cpp
# We need build-essential for C++ compilation 
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
COPY requirements.txt .
# If using CPU only, we might want to install llama-cpp-python usually. 
# Note: For GPU support, check llama-cpp-python docs for CMAKE_ARGS
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
