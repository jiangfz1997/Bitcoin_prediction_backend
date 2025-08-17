# Bitcoin Price Prediction Service

This project provides a lightweight Python application environment using Docker Compose. It includes dependency management and a basic database migration step.
## Quick Start

Follow the steps below to set up and run the project.

### 1. Start Docker services

```bash
docker-compose up -d
```
This command starts the containers defined in docker-compose.yml.

### 2. Install Python dependencies

Make sure you are inside the appropriate environment (container or virtualenv), then run:
```bash
pip install -r requirements.txt
```

### 3. Run the migration script
To launch the backend service responsible for real-time price prediction:
```bash
python manage.py run_predict_loop
```

To start the Django web server api, use the standard runserver command
```bash
python manage.py runserver 8000
```
For Grafana, you can access it at `http://localhost:3000` after starting the Docker services.
the grafana configuration file is located at `bitcoin-grafana.json` and can be imported into Grafana to visualize the Bitcoin price data.

### Files

docker-compose.yml – Defines the Docker service(s)

requirements.txt – Lists all Python dependencies

.env – Contains environment variables for the application

bitcoin-grafana.json - Grafana dashboard configuration file