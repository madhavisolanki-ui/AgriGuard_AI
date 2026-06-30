# AgriGuard

AgriGuard is a production-oriented Crop Health and Precision Agriculture System designed to help growers, agronomists, and operations teams detect plant stress early, monitor crop conditions, and support data-driven field decisions.

The repository is scaffolded for a FastAPI backend, AI model assets, reproducible data workflows, experiments, and containerized deployment. It is intentionally structured to scale from a prototype into a maintainable engineering system.

## Overview

AgriGuard combines computer vision, agronomic data processing, and API-driven delivery to support crop monitoring workflows such as:

- Crop health classification
- Disease and stress detection
- Image-based field diagnostics
- Data ingestion and preprocessing
- Model inference via API
- Dashboard-friendly backend services

The codebase is organized so engineering teams can separate concerns cleanly:

- `app/` contains the FastAPI application and service layer
- `models/` stores trained model artifacts and model metadata
- `data/` stores datasets, feature outputs, and intermediate artifacts
- `notebooks/` contains exploratory analysis and research notebooks
- `docker/` holds deployment helpers and container-related documentation
- `tests/` contains automated tests for API and service behavior

## Technical Highlights

- FastAPI-first backend architecture for high-performance HTTP APIs
- Modular application layout for maintainability and team ownership
- Clear separation between inference logic, API routes, and configuration
- Container-ready design with Python 3.10 support
- Test scaffold included from day one for regression protection
- Documentation-oriented repository structure suitable for production handoff

## Tech Stack

- Python 3.10
- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Pandas
- NumPy
- Streamlit
- python-dotenv

## System Architecture

```mermaid
flowchart LR
    A[User / Farm Operator] --> B[API Gateway / FastAPI]
    B --> C[Validation and Routing]
    C --> D[Inference Service]
    D --> E[Model Artifacts]
    D --> F[Preprocessing Pipeline]
    F --> G[Data Layer]
    B --> H[Monitoring / Logs]
    B --> I[Streamlit Dashboard]
```

> Diagram placeholder: replace this with the final architecture view once model serving, data storage, and observability choices are finalized.

## Repository Structure

```text
AgriGuard/
  app/
    api/
    core/
    schemas/
    services/
    main.py
  data/
  docker/
  models/
  notebooks/
  tests/
  Dockerfile
  LICENSE
  README.md
  requirements.txt
```

## Engineering Trade-offs

### 1. FastAPI over a heavier framework
FastAPI gives us speed, type hints, and a clean developer experience. The trade-off is that some enterprise features must be composed deliberately rather than inherited from a monolithic platform.

### 2. File-based model artifacts in the repo scaffold
Keeping `models/` as a dedicated directory makes local experimentation straightforward. The trade-off is that large production artifacts should eventually move to object storage or a model registry.

### 3. Lightweight initial dependency set
This scaffold intentionally starts with a focused dependency list. The trade-off is that observability, authentication, background jobs, and CI tooling can be introduced incrementally based on product maturity.

### 4. API and inference separated by design
Separating web routing from model logic improves testability and maintainability. The trade-off is a slightly larger initial code surface, which is worthwhile for long-term team scaling.

## Quality Strategy

- Unit tests for route and service behavior
- Input validation at the API boundary
- Explicit configuration management via environment variables
- Modular code that is easy to mock and extend
- Container-based runtime parity between development and production

## Local Development

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run the API with Uvicorn.

Example:

```bash
uvicorn app.main:app --reload
```

## Testing

Run the test suite with:

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t agriguard .
```

Run the container:

```bash
docker run --rm -p 8000:8000 agriguard
```

## Future Roadmap

- Add real crop disease inference pipelines
- Introduce experiment tracking and model versioning
- Add authentication and role-based access control
- Persist predictions and field observations in a database
- Add geospatial and satellite imagery support
- Add a Streamlit dashboard for agronomists and field teams
- Introduce CI/CD, linting, formatting, and release automation
- Add observability with structured logs, metrics, and tracing

## Contributing

This repository is structured to support collaborative development. Keep changes modular, documented, and test-backed. Prefer small, reviewable pull requests with clear scope.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
