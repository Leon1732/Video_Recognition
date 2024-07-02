import cv2
import random
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

sam_size = 8
pool_size = 3


train_labels = []
train_unit = []
predicted_labels = []
test_unit = []
predicted_class = []
n_components = 100
pca = PCA(n_components=n_components)
def feature_extract(frame_extract):
    frame_float = frame_extract.astype(np.float32)

    #计算均值和方差
    if i == 2:
        mean = [[0.07, 0.07, 0.07]]
        std = [[0.1, 0.09, 0.08]]

    else:
        # 计算均值和方差
        mean, std = cv2.meanStdDev(frame_float)
        mean = mean.T
        std = std.T
    normalized_frame = (frame_float - mean) / std

    # pool
    width = normalized_frame.shape[1] // pool_size
    height = normalized_frame.shape[0] // pool_size
    pooled_frame = cv2.resize(normalized_frame, (width, height),
                              interpolation=cv2.INTER_AREA)  # 调整帧大小，使用平均池化方式
    return pooled_frame

# 视频文件所在的文件夹路径
train_set = np.array(['EE6222 train and validate 2023/train/Jump',
                      'EE6222 train and validate 2023/train/Run',
                      'EE6222 train and validate 2023/train/Sit',
                      'EE6222 train and validate 2023/train/Stand',
                      'EE6222 train and validate 2023/train/Turn',
                      'EE6222 train and validate 2023/train/Walk'])

for i in range(6):
    # 获取文件夹中所有视频文件的路径
    video_files = [os.path.join(train_set[i], f) for f in os.listdir(train_set[i]) if f.endswith('.mp4')]

    #截取视频
    for j in video_files:
        # 打开视频文件
        action = cv2.VideoCapture(j)  # 替换为您的视频文件路径
        fusion_frames = np.zeros(25440)
        frame_total = int(action.get(cv2.CAP_PROP_FRAME_COUNT))

        random_numbers = sorted([random.randint(0, frame_total) for _ in range(frame_total // sam_size)])

        #截取帧
        for no_frame in range(0, frame_total, 8):
            frame_number = no_frame  # 读取的帧号
            u = 0
            # 设置视频的当前帧位置
            action.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # 读取指定帧
            ret, frame = action.read()

            # 如果成功读取帧
            if ret:
                # normalize

                pooled_frame = feature_extract(frame)
                pooled_frame = pooled_frame.flatten()
                fusion_frames += pooled_frame
                u += 1
            else:
                # 如果没有成功读取帧，退出循环q
                break
        train_unit.append(fusion_frames/u)
        new_element = i
        train_labels.append(new_element)
train_features = np.array(train_unit)
train_features = pca.fit_transform(train_features)
train_labels = np.array(train_labels)

svm_classifier = svm.SVC(kernel='linear')  # 选择线性核SVM，也可以选择其他核函数
svm_classifier.fit(train_features, train_labels)


test_set = 'EE6222 train and validate 2023/validate'  # 替换为您的视频文件路径
# 获取文件夹中的所有文件
test_video_files = [f for f in os.listdir(test_set) if os.path.isfile(os.path.join(test_set, f))]

# 读取视频文件
for test_video_file in test_video_files:
    video_path = os.path.join(test_set, test_video_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #截取帧
    for n in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        # 如果成功读取帧
        if ret:
            test_pooled_frame = feature_extract(frame)
            test_unit.append(test_pooled_frame.flatten())
        else:
            # 如果没有成功读取帧，退出循环q
            break
    test_features = np.array(test_unit)
    test_features = pca.transform(test_features)
    predicted_unit = svm_classifier.predict(test_features)
    predicted_labels.append(int(np.mean(predicted_unit)))
    test_unit = []
predicted_labels = np.array(predicted_labels)
test_labels = np.concatenate([
    np.repeat(0, 17),
    np.repeat(1, 15),
    np.repeat(2, 15),
    np.repeat(3, 16),
    np.repeat(4, 17),
    np.repeat(5, 16)
])



accuracy = accuracy_score(test_labels, predicted_labels)
print(f"分类精度：{accuracy * 100:.2f}%")