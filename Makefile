.PHONY: install test lint run docker-build docker-up clean train

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	ruff check . --select E,F,W,I --ignore E501

lint-fix:
	ruff check . --select E,F,W,I --ignore E501 --fix

type-check:
	mypy app/ --ignore-missing-imports --no-error-summary

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	python -c "from app.model import generate_synthetic_training_data, train; X,y=generate_synthetic_training_data(); m=train(X,y); print(m)"

docker-build:
	docker build -t talent-radar:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/ *.egg-info
