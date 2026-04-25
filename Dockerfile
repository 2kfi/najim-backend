FROM python:3.11-slim

ENV ROOT_PASSWORD=password

RUN mkdir -p /app

WORKDIR /app

RUN apt-get update

RUN apt-get install fish openssh-server nano -y --no-install-recommends  --no-install-suggests

RUN mkdir -p /var/run/sshd

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

COPY . /app

RUN pip install -r /app/requirements.txt --break-system-packages

RUN chsh -s /usr/bin/fish root

EXPOSE 22

EXPOSE 8080

EXPOSE 5000

CMD bash -c "echo \"root:$ROOT_PASSWORD\" | chpasswd && /usr/sbin/sshd -D & tail -f /dev/null"