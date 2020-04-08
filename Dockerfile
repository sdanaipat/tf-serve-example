FROM tensorflow/serving:latest-gpu
# FROM tensorflow/serving:latest

COPY models /opt/tfserve/models
COPY conf /opt/tfserve/conf

CMD ["tensorflow_model_server", "--model_config_file=/opt/tfserve/conf/model.config"]
