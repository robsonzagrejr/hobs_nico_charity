install:
	@pip3 install poetry
	@poetry install
	@poetry run aws configure

launch-api:
	@poetry run python application.py

zappa-deploy:
	@poetry run zappa deploy

zappa-update:
	@poetry run zappa update

zappa-undeploy:
	@poetry run	zappa undeploy

deploy:
	@. $$(poetry env info -p)/bin/activate;\
	gunicorn --bind 0.0.0.0:5010 wsgi:app
