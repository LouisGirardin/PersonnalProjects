import os
import numpy as np
from skimage import io, color, util, feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K

# Define the desired size for all images
desired_height = 256
desired_width = 256

# Define the folder path containing the images
folder_path = "/Users/louloules/LOCAL_DISK_PC/ML_Medical_Imaging/ProjectPerso/Project Images"

# Initialize empty lists for images and labels
images = []
labels = []

# Iterate through the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.png'):
        # Load the image
        image_path = os.path.join(folder_path, file_name)
        image = io.imread(image_path)
        
        # Convert RGBA to RGB format (discard alpha channel if it exists)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Resize the image to the desired size
        resized_image = resize(image, (desired_height, desired_width))
        
        # Pad the resized image if necessary to ensure it has the desired dimensions
        padded_image = np.pad(resized_image, ((0, desired_height - resized_image.shape[0]), 
                                               (0, desired_width - resized_image.shape[1]), 
                                               (0, 0)),
                                               mode='constant')
        
        # Extract the label from the filename
        label = 1 if 'YES' in file_name else 0
        
        # Append the padded image and label to the respective lists
        images.append(padded_image)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten images
num_samples, height, width, channels = images.shape
images_flattened = images.reshape(num_samples, height * width * channels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(min_samples_split=5, n_estimators=100)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Evaluate the classifier
train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = rf_classifier.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# CAM generation function
def generate_cam(model, img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Load pre-trained VGG16 model without fully connected layers
base_model = VGG16(weights='imagenet', include_top=False)

# Create a new model with CAM functionality
cam_model = tf.keras.models.Sequential()
cam_model.add(base_model)
cam_model.add(tf.keras.layers.GlobalAveragePooling2D())
cam_model.add(tf.keras.layers.Dense(2))  # 2 classes

# Compile the model
cam_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Iterate through test images to generate CAMs
for i in range(len(X_test)):
    img_path = 'test_image.png'  # Replace with the path to the test image
    heatmap = generate_cam(cam_model, img_path, target_size=(256, 256))
    plt.imshow(heatmap)
    plt.title("Class Activation Map")
    plt.show()
