FROM python:3
ADD object-discord.py /
ADD requirements.txt /
ADD MobileNetSSD_deploy.caffemodel /
ADD MobileNetSSD_deploy.prototxt.txt /
ADD emojis.json /
COPY .env .env
RUN apt update && apt install -y cmake
RUN apt install -y python3-opencv
RUN pip install -r requirements.txt
CMD [ "python3", "./object-discord.py" ]
