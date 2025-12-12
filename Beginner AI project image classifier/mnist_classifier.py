import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(f"Training data shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.title(f"Label: {train_labels[0]}")
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10)
model.save('mnist_model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\ntest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_image(i, prediction_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(prediction_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel(f"Prediction: {predicted_label} ({100*np.max(prediction_array):.2f}%)\nTrue Label: {true_label}", 
               color=color)
    
def plot_value_array(i,prediction_array):
    ax = plt.gca()

    ax.grid(False)
    
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    
    ax.set_yticks([])
    ax.set_ylim([0,1])
    
    thisplot = ax.bar(range(10), prediction_array, color = "#777777")

    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    thisplot[test_labels[i]].set_color('blue')

num_rows = 4
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i,predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i,predictions[i])

plt.tight_layout()
plt.show()