import os
import cv2
import numpy as np
import pandas as pd

# Load the trained model to classify signs
from tensorflow.keras.models import load_model
model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic sign classes
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

# Function to read the CSV file and get image paths and corresponding class IDs
def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    paths = data['Path'].tolist()
    class_ids = data['ClassId'].tolist()
    return paths, class_ids

# Load the image and classify it
def classify(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
    pred = model.predict(image)[0]
    pred_class_index = np.argmax(pred)
    sign = classes[pred_class_index + 1]
    confidence = pred[pred_class_index] * 100
    print("Predicted Sign: ", sign)
    print("Confidence: {:.2f}%".format(confidence))

# Path to the folders
base_dir = r"C:\Users\hp\PycharmProjects\pythonProject1"

# Path to the CSV files
meta_csv_file = os.path.join(base_dir, "Meta.csv")
test_csv_file = os.path.join(base_dir, "Test.csv")
train_csv_file = os.path.join(base_dir, "Train.csv")

# Example usage for Meta folder
meta_image_paths, meta_class_ids = read_csv(meta_csv_file)
for i in range(len(meta_image_paths)):
    image_path = os.path.join(base_dir, meta_image_paths[i])
    class_id = meta_class_ids[i]
    print("Image:", image_path)
    print("Class ID:", class_id)
    classify(image_path)
    print("---")

# Example usage for Test folder
test_image_paths, test_class_ids = read_csv(test_csv_file)
for i in range(len(test_image_paths)):
    image_path = os.path.join(base_dir, test_image_paths[i])
    class_id = test_class_ids[i]
    print("Image:", image_path)
    print("Class ID:", class_id)
    classify(image_path)
    print("---")

# Example usage for Train folder
train_image_paths, train_class_ids = read_csv(train_csv_file)
for i in range(len(train_image_paths)):
    image_path = os.path.join(base_dir, train_image_paths[i])
    class_id = train_class_ids[i]
    print("Image:", image_path)
    print("Class ID:", class_id)
    classify(image_path)
    print("---")
