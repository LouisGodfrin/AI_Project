import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


######Value########

max_epochs = 20
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

###################
# VGG Block
def vgg_block(num_convs, num_filters):
    block = tf.keras.models.Sequential()
    for _ in range(num_convs):
        block.add(tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='relu'))
        tf.keras.layers.BatchNormalization(),
    block.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
    tf.keras.layers.Dropout(0.6),  # Dropout layer
    return block

# Display some images with their labels
def plot_sample_images(images, labels, classes):
    plt.figure(figsize=(len(class_names),len(class_names)))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(f"Label: {classes[labels[i][0]]}")
        plt.axis('off')
    plt.show()

#Plot Class Distribution
def plot_class_distribution(labels, class_names):
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(class_names, counts)
    plt.title("Class Distribution in CIFAR-10 Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.show()

def plot_history(history, model_name, max_epochs):
    epochs_range = range(min(len(history.history['accuracy']), max_epochs))
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'][:max_epochs], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'][:max_epochs], label='Test Accuracy')
    plt.title(f'{model_name} - Model accuracy (first {max_epochs} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'][:max_epochs], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'][:max_epochs], label='Test Loss')
    plt.title(f'{model_name} - Model loss (first {max_epochs} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    
    plt.savefig(f'plot/{model_name}.png')
    plt.show()


# VGG1 avec optimisation (SGD)
def vgg1_with_optimization():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# Add Dropout and Batch Normalization to VGG2 architecture
def vgg2_with_regularization():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), 
        vgg_block(2, 64),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# Add Dropout and Batch Normalization to VGG3 architecture
def vgg3_with_regularization():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32),
        vgg_block(2, 64),
        vgg_block(2, 128),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# Compile model
def compile_and_train(model, optimizer, x_train, y_train_one_hot, x_test, y_test_one_hot, max_epochs):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_one_hot, epochs=max_epochs,batch_size = 64, validation_data=(x_test, y_test_one_hot))
    return history


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Normalize the images to the range [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, len(class_names))
y_test_one_hot = tf.keras.utils.to_categorical(y_test, len(class_names))

# Create models
model_vgg1 = vgg1_with_optimization()
model_vgg2 = vgg2_with_regularization()
model_vgg3 = vgg3_with_regularization()

# Optimizers to compare
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.05),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.0001)
}
optimizers2 = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.05),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.0001)
}
optimizers3 = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.05),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.0001)
}


# Train each model with different optimizers
history_vgg1_rmsprop = compile_and_train(model_vgg1, optimizers["RMSprop"], x_train, y_train_one_hot, x_test, y_test_one_hot, max_epochs)
history_vgg2_rmsprop = compile_and_train(model_vgg2, optimizers2["RMSprop"], x_train, y_train_one_hot, x_test, y_test_one_hot, max_epochs)
history_vgg3_rmsprop = compile_and_train(model_vgg3, optimizers3["RMSprop"], x_train, y_train_one_hot, x_test, y_test_one_hot, max_epochs)

# Visualize performance
plot_history(history_vgg1_rmsprop, 'VGG1 - RMSprop', max_epochs)
plot_history(history_vgg2_rmsprop, 'VGG2 - RMSprop', max_epochs)
plot_history(history_vgg3_rmsprop, 'VGG3 - RMSprop', max_epochs)

# Visualize class distribution
plot_class_distribution(y_train, class_names)

# Display sample images with labels
plot_sample_images(x_train, y_train, class_names)
