FROM public.ecr.aws/lambda/python:3.9

COPY Pipfile Pipfile.lock ${LAMBDA_TASK_ROOT}

RUN pip install pipenv && \
    pipenv requirements > requirements.txt && \
    pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.lambda_handler" ]
