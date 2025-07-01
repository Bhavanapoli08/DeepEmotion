import os

train_path = "data/train"
for category in os.listdir(train_path):
    category_path = os.path.join(train_path, category)
    print(f"{category}: {len(os.listdir(category_path))} images")
