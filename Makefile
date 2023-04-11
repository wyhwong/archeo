export DOCKER_BUILDKIT=1
port?=8888

build:
	docker build -t parentguess .

run:
	port=${port} docker-compose up parentguess

jupyter_up:
	port=${port} docker-compose up -d parentguess_jupyter

jupyter_down:
	port=${port} docker-compose kill parentguess_jupyter

clean:
	port=${port} docker-compose down --remove-orphans
