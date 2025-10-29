FROM python:3.8-slim-buster
# (python:3.8-slim-buster) It will take a base image from linux environment
WORKDIR /app
COPY . /app

# RUN apt update -y && apt install -y awscli
RUN pip install awscli

RUN pip install -r requirements.txt
CMD ["python3","application.py"]