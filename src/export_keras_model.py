from typing import Callable, Tuple
import tensorflow as tf
from tensorflow import keras


tf.compat.v1.disable_eager_execution()


class ServingInputReceiver:
    """
    A callable object that returns a
    `tf.estimator.export.ServingInputReceiver`
    object that provides a method to convert
    `image_bytes` input to model.input
    """
    def __init__(
        self, img_size: Tuple[int],
        preprocess_fn: Callable = None,
        input_name: str = "input_1",
    ):

        self.img_size = img_size
        self.preprocess_fn = preprocess_fn
        self.input_name = input_name

    def decode_img_bytes(self, img_b64: str) -> tf.Tensor:
        """
        Decodes a base64 encoded bytes and converts it to a Tensor.
        Args:
            img_bytes (str): base64 encoded bytes of an image file
        Returns:
            img (Tensor): a tensor of shape (width, height, 3)
        """
        img = tf.io.decode_image(
            img_b64,
            channels=3,
            dtype=tf.uint8,
            expand_animations=False
        )
        img = tf.image.resize(img, size=self.img_size)
        img = tf.ensure_shape(img, (*self.img_size, 3))
        img = tf.cast(img, tf.float32)
        return img

    def __call__(self) -> tf.estimator.export.ServingInputReceiver:
        # a placeholder for a batch of base64 string encoded of image bytes
        imgs_b64 = tf.compat.v1.placeholder(
            shape=(None,),
            dtype=tf.string,
            name="image_bytes")

        # apply self.decode_img_bytes() to a batch of image bytes (imgs_b64)
        imgs = tf.map_fn(
            self.decode_img_bytes,
            imgs_b64,
            dtype=tf.float32)

        # apply preprocess_fn if applicable
        if self.preprocess_fn:
            imgs = self.preprocess_fn(imgs)

        return tf.estimator.export.ServingInputReceiver(
            features={self.input_name: imgs},
            receiver_tensors={"image_bytes": imgs_b64}
        )


def main():
    print("[1/3] Loading Keras pretrained Xception model..",)
    model = keras.applications.Xception(weights="imagenet")
    model.compile(loss="categorical_crossentropy")
    print("done")

    print("[2/3] Creating tf.estimator..",)
    estimator_save_dir = "estimators/xception"
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=estimator_save_dir)
    print("done")

    print("[3/3] Exporting model..",)
    export_model_dir = "models/classification/xception/"
    serving_input_receiver = ServingInputReceiver(
        img_size=(299, 299),
        preprocess_fn=keras.applications.xception.preprocess_input,
        input_name="input_1")
    estimator.export_saved_model(
        export_dir_base=export_model_dir,
        serving_input_receiver_fn=serving_input_receiver)
    print("done")
    print("all done.")


if __name__ == "__main__":
    main()
