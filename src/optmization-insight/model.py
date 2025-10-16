import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes):
    """Builds a simple CNN model."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

class CustomModel(keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.CategoricalAccuracy(name="acc")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Second order gradients are computationally expensive and were commented out
        # in the original script's training loop. I am keeping it commented here.
        # second_order_grads = tape.gradient(grads, self.model.trainable_variables)
        second_order_grads = [tf.zeros_like(g) for g in grads] # Placeholder

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, y_pred)

        return {
            "train_loss": self.loss_tracker.result(),
            "train_acc": self.acc_tracker.result(),
            "grads": grads,
            "second_order_grads": second_order_grads,
            "weights": self.model.trainable_variables
        }

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y, y_pred)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, y_pred)
        return {"val_loss": self.loss_tracker.result(), "val_acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
