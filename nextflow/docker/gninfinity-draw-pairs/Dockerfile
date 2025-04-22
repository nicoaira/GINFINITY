# Dockerfile for gninfinity-draw-pairs
FROM python:3.9-slim

# Install system dependencies: Git, Maven, Java
RUN apt-get update && apt-get install -y git maven default-jdk-headless && rm -rf /var/lib/apt/lists/*

# Clone RNArtistCore and build
RUN git clone https://github.com/fjossinet/RNArtistCore.git /opt/RNArtistCore \
    && cd /opt/RNArtistCore \
    && mvn clean package

# Create wrapper script for 'rnartistcore' command
RUN printf '#!/usr/bin/env bash\nJAR=$(ls /opt/RNArtistCore/target/rnartistcore-*-SNAPSHOT-jar-with-dependencies.jar)\nexec java -jar "$JAR" "$@"\n' > /usr/local/bin/rnartistcore \
    && chmod +x /usr/local/bin/rnartistcore

# Set working directory for the app
WORKDIR /app

# Copy the draw_pairs CLI script
COPY nextflow/modules/draw_pairs.py ./draw_pairs.py

# pip install cairosvg
RUN pip install cairosvg
# No external Python packages needed; skip requirements.txt

# # Default entrypoint: run draw_pairs.py
# ENTRYPOINT ["python3", "draw_pairs.py"]
# # If no args, show help
# CMD ["--help"]