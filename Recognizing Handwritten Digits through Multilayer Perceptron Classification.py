# In this exercise, I'll be training a multi-layer perceptron (MLP) to classify handwritten digits sourced from the MNIST dataset.
# The dataset consists of 28x28 grayscale images depicting handwritten digits ranging from 0 to 9.
# My task will be to categorize each image into one of the ten classes, with each class corresponding to a specific digit.


# Step 1: Load the dataset

# Obtain the MNIST dataset using scikit-learn.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load data from https://www.openml.org/d/554
mnist = fetch_openml("mnist_784", version=1, return_X_y=False, parser="auto")

# Extract features and labels
X, y = mnist.data, mnist.target.astype(int)

# Split into train and test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Convert to numpy arrays and scale for the model
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 2: Create and train a Multi-Layer Perceptron

# Use sklearn.neural_network to build a Multi-Layer Perceptron in a single command and train it using a second command.
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load data from https://www.openml.org/d/554
mnist = fetch_openml("mnist_784", version=1, return_X_y=False, parser="auto")

# Extract features and labels
X, y = mnist.data, mnist.target.astype(int)

# Split into train and test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Convert to numpy arrays and scale for the model
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create an MLPClassifier object
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=100,  # Increase max_iter for better convergence
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

# Train the MLPClassifier
mlp.fit(X_train, y_train)

# Step 3: Evaluate the model on a hold-out dataset

# Assess the performance of the model on data that was not used for training (test set).
print(f"Training set score: {mlp.score(X_train, y_train)}")
print(f"Test set score: {mlp.score(X_test, y_test)}")

# Show the images, predictions, and original labels for 10 images
# Get the predictions for the test dataset
predictions = mlp.predict(X_test)

# Show the predictions in a grid
plt.figure(figsize=(8, 4))

for index, (image, prediction, label) in enumerate(zip(X_test[0:10], predictions[0:10], y_test[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)

    # Green if correct, red if incorrect
    fontcolor = "g" if prediction == label else "r"
    plt.title("Prediction: %i\n Label: %i" % (prediction, label), fontsize=10, color=fontcolor)
    
    plt.axis("off")  # hide axes

plt.show()

# Add this print statement to inspect the shape of the 'image' variable
print("Shape of image:", np.shape(image))
