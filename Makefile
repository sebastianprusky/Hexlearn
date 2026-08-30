.PHONY: api dashboard instrumented dataset baseline torch personal smoke-ml test

api:
	PYTHONPATH=service .venv/bin/uvicorn playlens_api.combined:app --host 127.0.0.1 --port 8000

dashboard:
	cd dashboard && ./node_modules/.bin/next dev --hostname 127.0.0.1 --port 3001

instrumented:
	python3 -m http.server 8001 --bind 127.0.0.1 --directory instrumentation/hextris

dataset:
	PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.build_dataset

baseline:
	PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.train_baseline

torch:
	PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.train_torch

personal:
	PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.train_personalization

smoke-ml:
	PYTHONPATH=service:ml .venv/bin/python -m playlens_ml.smoke_pipeline

test:
	PYTHONPYCACHEPREFIX=/tmp/playlens-pycache PYTHONPATH=service:ml .venv/bin/python -m unittest discover -s service/tests
	node --test extension/core.test.js
	cd dashboard && ./node_modules/.bin/eslint .
	cd dashboard && ./node_modules/.bin/next build
