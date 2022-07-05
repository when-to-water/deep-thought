FROM public.ecr.aws/lambda/python:3.9

COPY main.py Pipfile Pipfile.lock ${LAMBDA_TASK_ROOT}

RUN pip install pipenv && \
    pipenv requirements > requirements.txt && \
    pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "main.lambda_handler" ]
