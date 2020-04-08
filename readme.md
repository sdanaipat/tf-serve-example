# Minimal Example to Deploy Keras models using Tensorflow Serving
This guide aims to show you how to export a Keras model and serve it using Tensorflow Serving.

[read more](https://jokarubi.com/deploying-keras-models-using-tensorflow-serving/)

## Usage:
### 1. Export Keras model
```bash
python src/export_keras_model.py
```

### 2. Build and start TF Server image
```bash
docker build -t localhost/tf-server:latest .
```
```bash
docker run -p 8501:8501 \
           -v $(pwd)/models:/opt/tfserve/models \
           localhost/tf-server:latest
```
### 3. Test the TF Server
```bash
python src/rest_client.py --img samples/image.jpg
```
should output
```bash
[[('n02123045', 'tabby', 0.615710795),
  ('n02124075', 'Egyptian_cat', 0.177066833),
  ('n02123159', 'tiger_cat', 0.0166369341),
  ('n02909870', 'bucket', 0.00901011378),
  ('n02971356', 'carton', 0.00718645658)]]
```
