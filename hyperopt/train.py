import tensorflow as tf
from tensorflow.keras import datasets, layers, models
## hyperopt import
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

##hyper opt def search space 
space = {
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
}


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


def model_train(params):
	model.compile(optimizer=params['optimizer'],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	# hyperopt return
	return {'loss':-test_acc,'status':STATUS_OK}
## hyperopt start tuning
trials = Trials()
best = fmin(model_train, space, algo=tpe.suggest, max_evals=10, trials=trials)
print(best)

for trial in trials.trials[:2]:
    print(trial)
