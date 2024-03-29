import tensorflow as tf
from tensorflow.keras import datasets, layers, models
#optuna import
import optuna


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


def model_train(trial):
    #optuna def search space
    op = trial.suggest_categorical('op', [0.1,0.01,0.001,0.0001,0.00001])
    model.compile(optimizer=op,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    #optuna return
    return -test_acc
#optuna start tuning
study = optuna.create_study()
study.optimize(model_train, n_trials=5)
print(study.best_params)
print(study.best_value)
