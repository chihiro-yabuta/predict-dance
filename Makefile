data ?= "188JVmh_h903YI0FR7mgAKU2MVosFMjaD"
json ?= "1vPzsWAHS43LwtIzR4V12Y529kKFKrx5N"
CODE ?= "$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
dLb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${data}"
jLb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${json}"

.PHONY: zip, clean, down, setup, run, sw, dp, ds, jn, rm, cm

default:
	docker compose up -d
	make zip
	unzip data.zip
	unzip json.zip
	cp -r data/video ./
	cp -r data/archive ./
	rm -f -R __MACOSX data data.zip json.zip
	mkdir flow out test flow/video flow/json
	mkdir out/video out/video/cam out/video/edited out/video/removed
	mkdir out/img out/model out/src out/src/edited out/src/removed
zip:
	curl -Lb /tmp/cookie ${dLb} -o data.zip
	curl -Lb /tmp/cookie ${jLb} -o json.zip
clean:
	rm -f -R __MACOSX data data.zip json json.zip archive video flow out test
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
	make ds

sw:
	python monitor.py show
dp:
	python monitor.py dump
ds:
	python monitor.py dist
jn:
	python monitor.py json
rm:
	python monitor.py remove
cm:
	python monitor.py cam