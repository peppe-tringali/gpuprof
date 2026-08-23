# Publish workflow, safe by default. Each target is one thing.
# Assumes you have ~/.pypirc configured with tokens for `testpypi`
# and `pypi` (see README / `docs/publish.md`).

PYTHON ?= python
VENV ?= .venv
PYPI_INDEX_TEST := https://test.pypi.org/simple/
PYPI_INDEX_PROD := https://pypi.org/simple

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make test           run the unit + integration test suite"
	@echo "  make build          clean + build sdist + wheel + twine check"
	@echo "  make verify-wheel   install the built wheel in a scratch venv"
	@echo "  make publish-test   upload to TestPyPI (dry-run for real publish)"
	@echo "  make publish        upload to PyPI (the real thing)"
	@echo "  make tag            git tag v<version-from-pyproject>"
	@echo "  make release        test → build → verify → tag (no publish)"
	@echo "  make clean          rm dist/ build/ *.egg-info/"

.PHONY: test
test:
	$(VENV)/bin/pytest tests/ -ra --tb=short

.PHONY: clean
clean:
	rm -rf dist/ build/ src/*.egg-info/

.PHONY: build
build: clean
	$(VENV)/bin/pip install --quiet --upgrade build twine
	$(VENV)/bin/python -m build
	$(VENV)/bin/twine check dist/*
	@echo
	@echo "== wheel contents preview =="
	@unzip -l dist/*.whl | head -40

.PHONY: verify-wheel
verify-wheel:
	@rm -rf /tmp/gp-verify
	@python3 -m venv /tmp/gp-verify
	@WHL=$$(ls dist/*.whl | head -1); \
	 echo "Installing $$WHL[server,host]"; \
	 /tmp/gp-verify/bin/pip install --quiet "$$WHL[server,host]"
	/tmp/gp-verify/bin/gpuprof version
	/tmp/gp-verify/bin/gpuprof selfcheck | tail -12
	@rm -rf /tmp/gp-verify

.PHONY: publish-test
publish-test: build verify-wheel
	$(VENV)/bin/twine upload --repository testpypi dist/*
	@echo
	@echo "Verify with:"
	@echo "  pip install --index-url $(PYPI_INDEX_TEST) \\"
	@echo "              --extra-index-url $(PYPI_INDEX_PROD) gpuprof"

.PHONY: publish
publish: build verify-wheel
	@echo "Uploading to PyPI (real)..."
	@echo "You have 5 seconds to Ctrl-C if this is wrong."
	@sleep 5
	$(VENV)/bin/twine upload dist/*

.PHONY: tag
tag:
	@V=$$(grep '^version' pyproject.toml | head -1 | cut -d\" -f2); \
	echo "Tagging v$$V"; \
	git tag "v$$V" && git push --tags

.PHONY: release
release: test build verify-wheel tag
	@echo
	@echo "Ready to publish. Run:"
	@echo "  make publish-test    # to TestPyPI first (recommended)"
	@echo "  make publish         # to real PyPI"
