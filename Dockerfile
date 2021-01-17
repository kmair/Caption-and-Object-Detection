FROM tensorflow/tensorflow:2.3.1
COPY . /app 
WORKDIR /app
RUN pip install -r requirements.txt
