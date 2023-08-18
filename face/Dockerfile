FROM python:3
ADD face-discord.py /
ADD requirements.txt /
ADD shape_predictor_68_face_landmarks.dat /
RUN apt update && apt install -y cmake
RUN apt install -y python3-opencv
RUN pip install -r requirements.txt
COPY .env .env
CMD [ "python3", "./face-discord.py" ]
