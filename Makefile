.PHONY: test package clean unittest docs

PYTHON ?= $(shell which python)
PYTEST ?= $(shell which pytest)

PROJ_DIR  := ${PWD}
DOC_DIR   := ${PROJ_DIR}/docs
BUILD_DIR := ${PROJ_DIR}/build
DIST_DIR  := ${PROJ_DIR}/dist
TEST_DIR  := ${PROJ_DIR}/test
SRC_DIR   := ${PROJ_DIR}/digging
FORMAT_DIR ?= ${SRC_DIR}

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR}

test: unittest

unittest:
	$(PYTEST) "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

docs:
	$(MAKE) -C "${DOC_DIR}" html

format:
	yapf --in-place --recursive -p --verbose --style .style.yapf ${FORMAT_DIR}
format_test:
	bash format.sh ${FORMAT_DIR} --test
flake_check:
	flake8 ${FORMAT_DIR}
