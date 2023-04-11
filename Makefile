export DOCKER_BUILDKIT=1

port ?= 8888

build:
	docker build -t remnantguess .

run:
	docker-compose up remnantguess

jupyter_up:
	port=${port} docker-compose up -d remnantguess_jupyter

jupyter_down:
	port=${port} docker-compose kill remnantguess_jupyter

clean:
	docker-compose down --remove-orphans
