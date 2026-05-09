FROM python:3.11-slim

ENV ROOT_PASSWORD=password

RUN mkdir -p /app

WORKDIR /app

RUN apt-get update

RUN apt-get install fish openssh-server nano -y --no-install-recommends --no-install-suggests \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt --break-system-packages

RUN chsh -s /usr/bin/fish root

EXPOSE 22

EXPOSE 8080

CMD ["python", "app.py"]