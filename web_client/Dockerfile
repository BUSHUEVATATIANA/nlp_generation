FROM python:3.9
COPY ./app ./
# Копируем файл requirements.txt в рабочую директорию контейнера
COPY ./requirements.txt ./
# Копируем файл uwsgi.ini в рабочую директорию контейнера
# Запускаем установку необходимых зависимостей
RUN pip install -r requirements.txt && pip install --no-cache-dir torch
CMD [ "python", "./app.py" ]