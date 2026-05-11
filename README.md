# ML Model Monitoring Stack

Production-grade monitoring for deployed ML models. Detects data drift, concept drift, and prediction distribution shift. Auto-retrains on degradation.

## Stack

| Service     | Role                                      | Port |
|-------------|-------------------------------------------|------|
| MLflow      | Experiment tracking + model registry      | 5000 |
| Evidently   | Drift detection engine (custom service)   | 8000 |
| Prometheus  | Metrics storage + alerting engine         | 9090 |
| Grafana     | Live dashboards + alert notifications     | 3000 |

## Quick start

```bash
# 1. Copy env config
cp .env .env.local   # edit thresholds if needed

# 2. Spin up all services
docker compose up --build -d

# 3. Access UIs
open http://localhost:5000   # MLflow
open http://localhost:3000   # Grafana  (admin / admin)
open http://localhost:9090   # Prometheus
```

## Build order

1. **Phase 1** — this scaffold ✓
2. **Phase 2** — MLflow + train baseline model
3. **Phase 3** — Evidently drift detection service
4. **Phase 4** — Prometheus metrics exporter
5. **Phase 5** — Grafana dashboards + alerts
6. **Phase 6** — Auto-retraining trigger
7. **Phase 7** — End-to-end smoke test

## Folder structure

```
ml-monitoring/
├── docker-compose.yml
├── .env
├── monitoring/           # Evidently service + retrainer
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── monitor.py        # Phase 3
│   └── retrainer.py      # Phase 6
├── mlflow/
│   └── Dockerfile
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboards.yml
│   └── dashboards/       # JSON dashboards (Phase 5)
├── data/
│   ├── reference/        # Baseline dataset (Phase 2)
│   └── incoming/         # Live data window
├── models/               # Serialised model artifacts
└── scripts/
    └── train.py          # Phase 2
```
