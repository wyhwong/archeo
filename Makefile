export DOCKER_BUILDKIT=1

export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)
export TZ?=Asia/Hong_Kong

export VERSION?=devel
export PORT?=8888
export LOGLEVEL?=20
export MAX_WORKER?=20

build:
	docker-compose build

run:
	docker-compose up ancestral_bh

jupyter_up:
	docker-compose up -d ancestral_bh_jupyter

jupyter_down:
	docker-compose kill ancestral_bh_jupyter

clean:
	docker-compose down --remove-orphans
