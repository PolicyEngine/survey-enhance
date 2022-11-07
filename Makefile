format:
	black . -l 79

install:
	pip install -e .

all: install format
