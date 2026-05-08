FROM python:3.10-slim

WORKDIR /app

# Install standard web dependencies
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

# Force install the lightweight CPU version of PyTorch to speed up the build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the CMNet repo and your app.py into the container
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]