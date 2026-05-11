FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.12.1 \
    boto3==1.34.0 \
    psutil==5.9.8

RUN mkdir -p /mlflow/artifacts && chmod -R 777 /mlflow

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
