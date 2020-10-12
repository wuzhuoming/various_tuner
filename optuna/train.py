import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from ray import tune


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


def model_train(config):
	# optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
	model.compile(optimizer=config['optimizer'],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	tune.report(mean_loss= -test_acc)

config={
        'optimizer': tune.choice(['adam', 'rmsprop'])
    }

analysis = tune.run(
    model_train,
    config=config,
    num_samples=5,
    stop={"training_iteration": 5},
    local_dir="./results", 
    name="test_experiment"
    )

print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
print(df)
