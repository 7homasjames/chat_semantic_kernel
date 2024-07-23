FROM base-image

WORKDIR /app/
COPY . /app/
RUN chmod +x script.sh

RUN pip install fastapi

CMD ["./script.sh"]
