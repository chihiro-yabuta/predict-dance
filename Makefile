FILE_ID ?= "188JVmh_h903YI0FR7mgAKU2MVosFMjaD"
CODE ?= "$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
sc = "https://drive.google.com/uc?export=download&id=${FILE_ID}"
Lb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}"

.PHONY: zip, clean, down, show, dump, flow, remove, cam

default:
	docker compose up -d
	make zip
	unzip data.zip
	cp -r data/video ./
	cp -r data/archive ./
	rm -f -R __MACOSX data data.zip
	mkdir flow out test
	mkdir out/video out/video/cam out/video/edited out/video/removed
	mkdir out/model out/src out/src/edited out/src/removed
	make dump
zip:
	curl -sc /tmp/cookie ${sc} > /dev/null
	curl -Lb /tmp/cookie ${Lb} -o data.zip
clean:
	rm -f -R __MACOSX data data.zip archive video flow out test
down:
	docker compose down
	docker system prune -a

show:
	python monitor.py show
dump:
	python monitor.py dump
flow:
	python monitor.py flow
remove:
	python monitor.py remove
cam:
	python monitor.py cam