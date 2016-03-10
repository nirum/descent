all:
	python setup.py install

develop:
	python setup.py develop

test2:
	python2 /usr/local/bin/nosetests --logging-level=INFO

test3:
	nosetests -v --with-coverage --cover-package=descent --logging-level=INFO

clean:
	rm -rf htmlcov/
	rm -rf descent.egg-info
	rm -f descent/*.pyc
	rm -rf descent/__pycache__

upload:
	python setup.py sdist bdist_wininst upload
