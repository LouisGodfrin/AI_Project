import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# VGG Block
def vgg_block(num_convs, num_filters):
    block = tf.keras.models.Sequential()
    for _ in range(num_convs):
        block.add(tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same', activation='relu'))
    block.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
    return block

# VGG1 - 1 VGG Block
def vgg1():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), # 1 VGG block
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# VGG2 - 2 VGG Blocks
def vgg2():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), # 1st VGG block
        vgg_block(2, 64), # 2nd VGG block
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# VGG3 - 3 VGG Blocks
def vgg3():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), # 1st VGG block
        vgg_block(2, 64), # 2nd VGG block
        vgg_block(2, 128), # 3rd VGG block
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Display some images with their labels
def plot_sample_images(images, labels, classes):
    plt.figure(figsize=(10,10))
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

def plot_history(history, model_name, max_epochs=5):
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
    
    plt.show()

# Add Dropout and Batch Normalization to VGG2 architecture
def vgg2_with_regularization():
    model = tf.keras.models.Sequential([
        vgg_block(2, 32), 
        tf.keras.layers.Dropout(0.5),  # Dropout layer
        vgg_block(2, 64),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Compile model
def compile_and_train(model, optimizer, x_train, y_train_one_hot, x_test, y_test_one_hot, epochs=10):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_one_hot, epochs=epochs, validation_data=(x_test, y_test_one_hot), batch_size=64)
    return history


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 has 10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Normalize the images to the range [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

# Create models
model_vgg1 = vgg1()
model_vgg2 = vgg2()
model_vgg3 = vgg3()

# Optimizers to compare
optimizers = {
    "SGD": tf.keras.optimizers.SGD(),
    "Adam": tf.keras.optimizers.Adam(),
    "RMSprop": tf.keras.optimizers.RMSprop()
}

# Train each model with different optimizers
history_vgg1_sgd = compile_and_train(model_vgg1, optimizers["SGD"], x_train, y_train_one_hot, x_test, y_test_one_hot)
history_vgg2_adam = compile_and_train(model_vgg2, optimizers["Adam"], x_train, y_train_one_hot, x_test, y_test_one_hot)
history_vgg3_rmsprop = compile_and_train(model_vgg3, optimizers["RMSprop"], x_train, y_train_one_hot, x_test, y_test_one_hot)

# Visualize performance
plot_history(history_vgg1_sgd, 'VGG1 - SGD')
plot_history(history_vgg2_adam, 'VGG2 - Adam')
plot_history(history_vgg3_rmsprop, 'VGG3 - RMSprop')

# Train VGG2 with regularization
model_vgg2_reg = vgg2_with_regularization()
history_vgg2_reg_adam = compile_and_train(model_vgg2_reg, optimizers["Adam"], x_train, y_train_one_hot, x_test, y_test_one_hot)

# Use the function with the optimizer classes
history_vgg2_reg_adam = compile_and_train(model_vgg2_reg, tf.keras.optimizers.Adam, x_train, y_train_one_hot, x_test, y_test_one_hot)

# Visualize performance with regularization
plot_history(history_vgg2_reg_adam, 'VGG2 with Regularization - Adam')

# Visualize class distribution
plot_class_distribution(y_train, class_names)

# Display sample images with labels
plot_sample_images(x_train, y_train, class_names)
