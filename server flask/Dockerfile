
FROM python:3.11.12


WORKDIR /app


COPY  requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server_phar.py"]