import os
import pandas as pd


TRAIN_DIR = "data\\train"
TEST_DIR = "data\\test"


def count_images_in_folder(folder_path):
    """
    Count image files inside each class folder.
    """

    valid_extensions = (".jpg", ".jpeg", ".png")

    class_counts = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)

        if os.path.isdir(class_path):
            image_count = 0

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(valid_extensions):
                    image_count += 1

            class_counts[class_name] = image_count

    return class_counts


def main():
    train_counts = count_images_in_folder(TRAIN_DIR)
    test_counts = count_images_in_folder(TEST_DIR)

    print("\n==============================")
    print("Dataset Summary")
    print("==============================")

    print("Total train classes:", len(train_counts))
    print("Total test classes:", len(test_counts))

    print("Total train images:", sum(train_counts.values()))
    print("Total test images:", sum(test_counts.values()))

    print("\nFirst 10 train classes:")
    for class_name, count in list(train_counts.items())[:10]:
        print(class_name, ":", count)

    df = pd.DataFrame({
        "class_name": list(train_counts.keys()),
        "train_count": list(train_counts.values()),
        "test_count": [test_counts.get(cls, 0) for cls in train_counts.keys()]
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset_summary.csv", index=False)

    print("\nDataset summary saved at:")
    print("data/dataset_summary.csv")


if __name__ == "__main__":
    main()