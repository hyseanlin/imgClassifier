import argparse
import os
import glob
import csv

arg_parser = argparse.ArgumentParser(description='獲取影像資料。')
arg_parser.add_argument(
    '--data-type',
    help='資料的存放目錄',
    default='train_data',
)
args = arg_parser.parse_args()

path = args.data_type
labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

annotations = dict()
for label in labels:
    print(f'Label: {label}')
    image_files = glob.glob(os.path.join(path, label, '*.jpg'))  # adjust the file extension if needed
    annotations[label] = image_files
    for image_file in image_files:
        print(image_file)

annotations_path = os.path.join(path, 'annotations.csv')
with open(annotations_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image File", "Label Index", "Label Name"])
    for idx, label in enumerate(annotations):
        for img_path in annotations[label]:
            writer.writerow([img_path, idx, label])