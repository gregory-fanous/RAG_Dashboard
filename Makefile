.PHONY: test run-backend run-frontend docker-up docker-down run-method-sweep consolidate-method-sweeps download-datasets

test:
	pytest

run-backend:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev

docker-up:
	docker compose up --build

docker-down:
	docker compose down

run-method-sweep:
	PYTHONPATH=. python scripts/run_backend_method_sweep.py

consolidate-method-sweeps:
	PYTHONPATH=. python scripts/consolidate_method_sweeps.py

download-datasets:
	PYTHONPATH=. python scripts/download_hf_datasets.py
