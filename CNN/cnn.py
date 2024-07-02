import cv2
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_video_files_in_folder(folder_path):
    video_files = []
    video_extensions = ['.mp4']

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(file_path)
            if file_extension.lower() in video_extensions:
                video_files.append(file_path)

    return video_files

def extract_videos_from_folder(folder_path):
    video_paths = [os.path.join(folder_path, vid) for vid in sorted(os.listdir(folder_path)) if vid.endswith(('.mp4'))]
    return video_paths

# Prepare training set and test set
# path of folder
folder_path_train = 'your_training_videos'
folder_path_test = 'your_validation_videos'
train_videos = extract_videos_from_folder(folder_path_train)
print(train_videos)
test_videos = extract_videos_from_folder(folder_path_test)
print(test_videos)
train_labels = np.concatenate([
        np.repeat(0, 25),
        np.repeat(1, 25),
        np.repeat(2, 25),
        np.repeat(3, 25),
        np.repeat(4, 25),
        np.repeat(5, 25)
])
test_labels = np.concatenate([
        np.repeat(1, 15),
        np.repeat(2, 15),
        np.repeat(3, 16),
        np.repeat(4, 17),
        np.repeat(5, 16),
        np.repeat(0, 17)
])

# parameter setting
sample_rate = 5
merge_count = 12

# ResNet model
model = models.resnet50(pretrained=True)
model = model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# normalize
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# extract features
def extract_features(frame):
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    return output.squeeze().numpy()

# sampling
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % sample_rate == 0:
            img_array = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feature = extract_features(frame)
            features.append(feature)
    cap.release()
    return features

train_features = [np.mean(process_video(video_path), axis=0) for video_path in train_videos]
test_features = [np.mean(process_video(video_path), axis=0) for video_path in test_videos]

# classification
classifier = SVC(kernel='linear')
train_features_flattened = np.reshape(train_features, (len(train_features), -1))
test_features_flattened = np.reshape(test_features, (len(test_features), -1))
classifier.fit(train_features_flattened, train_labels)

# test accuracy
predictions = classifier.predict(test_features_flattened)
accuracy = accuracy_score(test_labels, predictions)
print(predictions)
print(f"Accuracy using SVM: {accuracy}")

















