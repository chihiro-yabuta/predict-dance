data ?= "188JVmh_h903YI0FR7mgAKU2MVosFMjaD"
json ?= "1vPzsWAHS43LwtIzR4V12Y529kKFKrx5N"
CODE ?= "$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
dLb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${data}"
jLb = "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${json}"

.PHONY: zip, clean, up, down, code, run, sw, dp, ds, jn, rm, ccm, tcm

default:
	make zip
	unzip data.zip
	unzip json.zip
	cp -r data/video ./
	cp -r data/archive ./
	rm -f -R __MACOSX data data.zip json.zip
	mkdir out test out/flow out/model out/src out/video
	mkdir out/video/cam out/video/edited out/video/removed
	mkdir out/src/edited out/src/removed
	make dp
zip:
	curl -Lb /tmp/cookie ${dLb} -o data.zip
	curl -Lb /tmp/cookie ${jLb} -o json.zip
clean:
	rm -f -R __MACOSX data data.zip json json.zip archive video out test
up:
	docker compose up -d
down:
	docker compose down
	docker system prune -a

code:
	code --install-extension ms-vscode-remote.remote-containers
	code --install-extension ms-azuretools.vscode-docker
	code --install-extension ms-python.python
	code --install-extension ms-python.vscode-pylance
	code --install-extension shardulm94.trailing-spaces
run:
	python main.py
	make ccm
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
ccm:
	python monitor.py conv_cam
tcm:
	python monitor.py trans_cam