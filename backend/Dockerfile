FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Give permission to script
RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]