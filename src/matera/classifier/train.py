import argparse
import csv
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score

from matera.classifier.model import AmbiguityClassifier


def main():
    parser = argparse.ArgumentParser(description="Train the Ambiguity Classifier")
    parser.add_argument("--dataset", type=str, default="data/training/ambiguous_rois.csv")
    parser.add_argument("--output", type=str, default="models/matera-pre-v1/classifier.joblib")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found.")
        return

    features = [
        "dark_pixel_ratio",
        "foreground_area_ratio",
        "contour_count",
        "largest_component_ratio",
        "bbox_fill_ratio",
    ]

    # Load data
    train_X, train_y = [], []
    dev_X, dev_y = [], []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page_num = int(row["page_number"])
            expected = int(row["expected_mark"])
            
            x = [float(row[f]) for f in features]
            
            # Split by page (1-5 for train, 6-10 for dev)
            if page_num <= 5:
                train_X.append(x)
                train_y.append(expected)
            else:
                dev_X.append(x)
                dev_y.append(expected)

    print(f"Loaded {len(train_y)} training samples (Pages 1-5)")
    print(f"Loaded {len(dev_y)} dev samples (Pages 6-10)")
    
    if not train_X:
        print("Not enough data to train. Exiting.")
        return
        
    # Train
    print("Training model...")
    clf = AmbiguityClassifier()
    clf.train(train_X, train_y)
    
    # Feature Importance
    importances = clf.get_feature_importance(features)
    print("\n--- Feature Importance ---")
    for feat, coef in importances.items():
        print(f"{feat}: {coef:.4f}")
        
    # Evaluate
    if dev_X:
        print("\n--- Evaluation on Dev Set (Pages 6-10) ---")
        preds = clf.predict(dev_X)
        print(f"Accuracy: {accuracy_score(dev_y, preds):.4f}")
        print(classification_report(dev_y, preds, zero_division=0))
        
    # Save
    clf.save(args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
