FROM mirror.gcr.io/library/python:3.11
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY src /app/src
RUN pip install -r /app/requirements.txt
ENTRYPOINT ["bash"]
