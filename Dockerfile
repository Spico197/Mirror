FROM python:3.10-slim

#Install git
RUN apt-get update
RUN apt-get -y install git

# prepare scripts
WORKDIR /app/
# copy all files to workdir
COPY ./ /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# start service
EXPOSE 7860
CMD ["python", "-m", "src.app.api_backend"]
