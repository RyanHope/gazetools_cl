all:
	python setup.py build sdist

install:
	python setup.py install

clean:
	rm -rf dist build gazetools.egg* MANIFEST
