from keras.callbacks import EarlyStopping
from dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
from utils.consts import CORRUPTIONS_TYPES
from sklearn.metrics import f1_score
import numpy as np
from utils.metrics import write_fscore_result
from keras.backend import clear_session

KFOLD_N_SPLITS = 10


def experiment(model, data_augmentation_layers):
    x, y, splits = get_cifar10_kfold_splits(KFOLD_N_SPLITS)

    for fold_number, (train_index, val_index) in splits:
        clear_session()

        x_train_fold, y_train_fold = x[train_index], y[train_index]
        x_val_fold, y_val_fold = x[val_index], y[val_index]

        train_ds = get_cifar10_dataset(x_train_fold, y_train_fold, data_augmentation_layers)
        val_ds = get_cifar10_dataset(x_val_fold, y_val_fold)

        model.fit(
            train_ds,
            val_dataset=val_ds,
            epochs=100,
            callbacks=[
                EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)
            ],
            fold_number=fold_number,
        )

        for corruption in CORRUPTIONS_TYPES:
            corrupted_dataset = get_cifar10_corrupted(corruption)
            corrupted_labels = np.concatenate([y for x, y in corrupted_dataset], axis=0)
            predictions = model.predict(corrupted_dataset)
            predicted_labels = np.argmax(predictions, axis=1)
            f1 = f1_score(corrupted_labels, predicted_labels, average='macro')

            write_fscore_result(
                corruption,
                model.approach_name,
                model.name,
                f1,
            )
