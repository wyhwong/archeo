export DOCKER_BUILDKIT=1

export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)
export TZ?=Asia/Hong_Kong

export VERSION?=devel
export PORT?=8888
export LOGLEVEL?=20

build:
	mkdir -p ./results
	docker-compose build

run:
	docker-compose up paper

jupyter_up:
	docker-compose up -d paper_jupyter

jupyter_down:
	docker-compose kill paper_jupyter

clean:
	docker-compose down --remove-orphans
