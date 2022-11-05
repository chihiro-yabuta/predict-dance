FILE_ID ?= "188JVmh_h903YI0FR7mgAKU2MVosFMjaD"
CODE ?= "$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
sc = "https://drive.google.com/uc?export=download&id=${FILE_ID}"
Lb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}"

.PHONY: zip, clean, down, setup, run, view, a, sw, dp, fw, rm, cm

default:
	docker compose up -d
	make zip
	unzip data.zip
	cp -r data/video ./
	cp -r data/archive ./
	rm -f -R __MACOSX data data.zip
	mkdir flow out test
	mkdir out/video out/video/cam out/video/edited out/video/removed
	mkdir out/img out/model out/src out/src/edited out/src/removed
zip:
	curl -sc /tmp/cookie ${sc} > /dev/null
	curl -Lb /tmp/cookie ${Lb} -o data.zip
clean:
	rm -f -R __MACOSX data data.zip archive video flow out test
down:
	docker compose down
	docker system prune -a

setup:
	code --install-extension ms-vscode-remote.remote-containers
	code --install-extension ms-azuretools.vscode-docker
	code --install-extension ms-python.python
	code --install-extension ms-python.vscode-pylance
	code --install-extension shardulm94.trailing-spaces
	make dp
run:
	python main.py
	make fw
view:
	make a -J

a:
	make fw
	make rm
	make cm
sw:
	python monitor.py show
dp:
	python monitor.py dump
fw:
	python monitor.py flow
rm:
	python monitor.py remove
cm:
	python monitor.py cam