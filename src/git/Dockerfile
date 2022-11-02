FROM python:3.9.15

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN pip install --upgrade pip wheel setuptools

COPY src/git/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt