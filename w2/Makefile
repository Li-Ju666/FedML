all: image
	docker run --rm -it -d -v $(CURDIR)/script:/srv keras:cpu
image: Dockerfile
	docker build -t keras:cpu .
