install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

train:
	python -m src.models.train

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete