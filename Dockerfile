# syntax=docker/dockerfile:1

FROM python:3.12.2

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD python3 scrape.py ; python3 app.py