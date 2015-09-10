all: dist

dist:
	python setup.py sdist

install: dist
	pip install dist/gazetools-*.tar.gz

clean:
	rm -rf dist build gazetools.egg* MANIFEST
