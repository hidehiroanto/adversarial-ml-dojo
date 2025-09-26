# syntax=docker/dockerfile:labs

FROM python:3.6.6-slim

WORKDIR /challenges
ADD requirements.txt .
RUN pip install -v --disable-pip-version-check --no-cache-dir -r requirements.txt
