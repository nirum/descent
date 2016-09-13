all:
	python setup.py install

develop:
	python setup.py develop

test:
	pytest --cov=descent --cov-report=html tests/

clean:
	rm -rf htmlcov/
	rm -rf descent.egg-info
	rm -f descent/*.pyc
	rm -rf descent/__pycache__

upload:
	python setup.py sdist bdist_wininst upload
