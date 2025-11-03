# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and code
COPY main.py .

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn

# Run the script
CMD ["python", "main.py"]
