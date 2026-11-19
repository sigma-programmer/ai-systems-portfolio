PY ?= .venv/bin/python
export AIKIT_OFFLINE ?= 1

.PHONY: setup test demos bench site dashboard snapshot clean

setup:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest

demos:
	@for d in projects/p*/demo.py; do echo "\n===== $$d ====="; $(PY) $$d || exit 1; done

bench:
	cd projects/p12_architecture_teardown && ../../$(PY) -m teardown bench --repeats 2

site:
	cd projects/p12_architecture_teardown && ../../$(PY) -m teardown site --single site/teardown.html

dashboard:
	$(PY) dashboard/server.py

snapshot:
	$(PY) dashboard/server.py --snapshot dashboard/snapshot.html

clean:
	find . -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache projects/p12_architecture_teardown/site
