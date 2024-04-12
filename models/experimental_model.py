import numpy as np
from utils.metrics import plot_history, write_evaluation_result, save_confusion_matrix


class ExperimentalModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, approach_name=''):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.approach_name = approach_name
        self.model = self._build_model()
        self.compile()

    def _build_model(self):
        return

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        self.model.summary()

    def fit(self, train_dataset, val_dataset=None, callbacks=None, batch_size=32, epochs=10, fold_number=0, save_history=False):
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        if save_history:
            plot_history(self.execution_name, history.history, fold_number)

        return history

    def evaluate(self, eval_ds, corruption_type, execution_name):
        loss, acc = self.model.evaluate(eval_ds)

        write_evaluation_result(corruption_type, execution_name, loss, acc)

        return loss, acc

    def predict(self, dataset):
        return self.model.predict(dataset)
