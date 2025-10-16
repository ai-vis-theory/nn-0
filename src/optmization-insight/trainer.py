import tensorflow as tf
from . import config
from .data_loader import get_data_generators
from .model import CustomModel, build_model
from .callbacks import OptimizationInsightCallback

class Trainer:
    def __init__(self):
        self.train_generator, self.test_generator = get_data_generators()
        self.model = self._build_and_compile_model()
        self.callbacks = self._setup_callbacks()

    def _build_and_compile_model(self):
        base_model = build_model(config.INPUT_SHAPE, config.NUM_CLASSES)
        model = CustomModel(base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss_fn=tf.keras.losses.CategoricalCrossentropy()
        )
        return model

    def _setup_callbacks(self):
        insight_callback = OptimizationInsightCallback(
            train_generator=self.train_generator
        )
        # The LR scheduler from the original code was returning the same lr
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        return [insight_callback]

    def train(self):
        self.model.fit(
            self.train_generator,
            epochs=config.EPOCHS,
            validation_data=self.test_generator,
            callbacks=self.callbacks,
            steps_per_epoch=len(self.train_generator),
            validation_steps=len(self.test_generator)
        )
