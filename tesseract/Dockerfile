FROM python:3
ADD tesseract-discord.py /
ADD requirements.txt /
RUN apt update && apt install -y tesseract-ocr
RUN apt install -y cmake
RUN apt install -y python3-opencv
RUN pip install -r requirements.txt
COPY .env .env
CMD [ "python3", "./tesseract-discord.py" ]
