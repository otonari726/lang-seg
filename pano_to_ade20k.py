import os
import cv2

def main(source_dir, target_dir):
    train_ids = []
    val_ids = []

    with open(os.path.join(source_dir, "splits", "train.txt")) as f:
        for line in f:
            line = line.rstrip()
            train_ids.append(line)

    with open(os.path.join(source_dir, "splits", "val.txt")) as f:
        for line in f:
            line = line.rstrip()
            val_ids.append(line)

    for train_id in train_ids:
        print("train", train_id)
        rgb = cv2.imread(os.path.join(source_dir, "img", train_id+".jpg"))
        ann = cv2.imread(os.path.join(source_dir, "ann", train_id+".png"), 0)
        cv2.imwrite(os.path.join(target_dir, "images", "training", train_id+".jpg"), rgb)
        cv2.imwrite(os.path.join(target_dir, "annotations", "training", train_id+".png"), ann)

    for val_id in val_ids:
        print("val", val_id)
        rgb = cv2.imread(os.path.join(source_dir, "img", val_id+".jpg"))
        ann = cv2.imread(os.path.join(source_dir, "ann", val_id+".png"), 0)
        cv2.imwrite(os.path.join(target_dir, "images", "validation", val_id+".jpg"), rgb)
        cv2.imwrite(os.path.join(target_dir, "annotations", "validation", val_id+".png"), ann)


if __name__ == "__main__":
    source_dir = "./datasets/pano_segmentation_dataset"
    target_dir = "./datasets/pano_segmentation_dataset"

    os.makedirs(os.path.join(target_dir, "images", "training"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images", "validation"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "annotations", "training"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "annotations", "validation"), exist_ok=True)

    main(source_dir, target_dir)
