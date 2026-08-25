import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = Path("data/dataset")
TRUE_MARKS_DIR = DATASET_DIR / "true_marks"
NOISE_DIR = DATASET_DIR / "noise"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (24, 24)

def extract_hog_features(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    # Tính HOG
    features = hog(
        img, 
        orientations=8, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False, 
        feature_vector=True
    )
    return features

def parse_page_num(filename):
    # p10_Q14_3_393px.png -> "p10" -> 10
    parts = filename.split('_')
    return int(parts[0][1:])

def main():
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("Loading True Marks...")
    for f in TRUE_MARKS_DIR.glob("*.png"):
        feat = extract_hog_features(f)
        p_num = parse_page_num(f.name)
        if p_num <= 7:
            X_train.append(feat)
            y_train.append(1)
        else:
            X_test.append(feat)
            y_test.append(1)
            
    print("Loading Noise...")
    for f in NOISE_DIR.glob("*.png"):
        feat = extract_hog_features(f)
        p_num = parse_page_num(f.name)
        if p_num <= 7:
            X_train.append(feat)
            y_train.append(0)
        else:
            X_test.append(feat)
            y_test.append(0)
            
    print(f"Train Size: {len(X_train)} (TP: {sum(y_train)}, FP: {len(y_train)-sum(y_train)})")
    print(f"Test Size:  {len(X_test)} (TP: {sum(y_test)}, FP: {len(y_test)-sum(y_test)})")
    
    # Huấn luyện SVM
    print("\nTraining SVM Classifier...")
    clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    # Đánh giá
    print("\nEvaluating on Test Set (Page 8-10)...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Noise (0)", "True Mark (1)"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # In ra các lỗi trên Test Set để xem nó có sai ở Page 10 Q14 Option 2 / Option 3 không
    print("\nAnalyzing Test Errors...")
    noise_paths = [f for f in NOISE_DIR.glob("*.png") if parse_page_num(f.name) > 7]
    true_paths = [f for f in TRUE_MARKS_DIR.glob("*.png") if parse_page_num(f.name) > 7]
    all_test_paths = true_paths + noise_paths
    
    for i, path in enumerate(all_test_paths):
        true_label = y_test[i]
        pred_label = y_pred[i]
        prob = clf.predict_proba([X_test[i]])[0]
        if true_label != pred_label:
            print(f"ERROR: {path.name} | True: {true_label} | Pred: {pred_label} | Prob: {prob}")
        elif "p10_Q14" in path.name:
            # Luôn in ra kết quả của P10 Q14 để kiểm tra đặc biệt
            print(f"CHECK P10 Q14: {path.name} | True: {true_label} | Pred: {pred_label} | Prob: {prob}")
            
    # Lưu mô hình
    model_path = MODEL_DIR / "shape_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
