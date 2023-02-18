all: build

format:
	black . -l 79

install:
	pip install -e .[dev]

build: install
	pip install wheel
	python setup.py sdist bdist_wheel

documentation:
	jb clean docs
	jb build docs
