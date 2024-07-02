import cv2
import numpy as np
import os
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 设置参数
sample_interval = 5  # 每隔多少帧采样一次
mhi_duration = 4      # MHI的时间跨度
svm_kernel = 'linear' # SVM的核函数类型


# 读取视频并提取MHI特征
def compute_MHI(video_path):
    average_motion = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    prev_frame = np.zeros((h, w), dtype=np.uint8)
    motion_history = np.zeros((h, w), dtype=np.float32)
    sum_motion = np.zeros((h, w))
    j = 1
    random_numbers = random.randint(1, sample_interval)
    while ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame_gray, prev_frame)
        _, motion_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, mhi_duration)
        prev_frame = frame_gray.copy()
        sum_motion += motion_history
        j += 1
        if j % 3 == 0:
            average_motion.append(sum_motion//3)
        for i in range(random_numbers):
            ret, frame = cap.read()
            if not ret:
                break
    j = 1
    transfer = []
    average_motion = np.array(average_motion)
    for t in range(len(average_motion)):
        transfer.append(average_motion[t].flatten())
    transfer_array = np.array(transfer)
    return transfer_array, t

# 从多个二级文件夹中提取视频路径
def extract_videos_from_folders(root_dir):
    video_paths = []
    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            video_paths.extend(sorted([os.path.join(subdir_path, vid) for vid in os.listdir(subdir_path) if vid.endswith(('.mp4', '.avi', '.mov'))]))
    return video_paths

# 从文件夹中提取视频路径
def extract_videos_from_folder(folder_path):
    video_paths = [os.path.join(folder_path, vid) for vid in sorted(os.listdir(folder_path)) if vid.endswith(('.mp4', '.avi', '.mov'))]
    return video_paths

# 加载训练数据和标签
def load_data():
    second_root_dir = 'EE6222 train and validate 2023/train'  # 替换为包含多个二级文件夹的根文件夹路径
    # 提取所有视频的路径并按顺序排列
    train_videos = extract_videos_from_folders(second_root_dir)
    train_labels = np.concatenate([
        np.repeat(0, 25),
        np.repeat(1, 25),
        np.repeat(2, 25),
        np.repeat(3, 25),
        np.repeat(4, 25),
        np.repeat(5, 25)
    ])
    X_train = []
    y_train = []
    k = 0
    for video, label in zip(train_videos, train_labels):
        mhi, number = compute_MHI(video)
        if k == 0:
            X_train = mhi
            k +=1
            y_train = np.full(number + 1, label)
        else:
            X_train = np.concatenate((X_train, mhi), axis=0)
            y_train = np.concatenate((y_train, np.full(number + 1, label)))
    k = 0
    return np.array(X_train), np.array(y_train)

# 加载测试数据和标签
def load_test_data():
    folder_path = 'EE6222 train and validate 2023/validate'  # 替换为包含多个视频的文件夹路径
    test_videos = extract_videos_from_folder(folder_path)
    test_labels = np.concatenate([
        np.repeat(0, 17),
        np.repeat(1, 15),
        np.repeat(2, 15),
        np.repeat(3, 16),
        np.repeat(4, 17),
        np.repeat(5, 16)
    ])
    X_test = []
    y_test = []
    frame_per_video = []
    k = 0
    for video, label in zip(test_videos, test_labels):
        mhi, number = compute_MHI(video)
        if k == 0:
            X_test = mhi
            k += 1
            y_test = np.full(number + 1, label)
        else:
            X_test = np.concatenate((X_test, mhi), axis=0)
            y_test = np.concatenate((y_test, np.full(number + 1, label)))
        frame_per_video.append(number)
    k = 0
    return np.array(X_test), test_labels, np.array(frame_per_video)

# 训练SVM模型
def train_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

# 测试SVM模型
def test_svm(svm, X_test, y_test, number):
    k = 0
    sum_y_pred = []
    y_pred = svm.predict(X_test)
    for i in number:
        sum_y_pred.append(int(np.mean(y_pred[k:k+i])))
        k = k+i
    sum_y_pred = np.array(sum_y_pred)
    print(sum_y_pred)
    accuracy = accuracy_score(y_test, sum_y_pred)
    return accuracy

if __name__ == "__main__":
    # 加载训练数据和测试数据
    X_train, y_train = load_data()
    X_test, y_test, frames_per_video = load_test_data()

    # 训练SVM模型
    svm_model = train_svm(X_train, y_train)

    # 测试SVM模型
    accuracy = test_svm(svm_model, X_test, y_test, frames_per_video)
    print("Accuracy:", accuracy)