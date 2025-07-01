import tensorflow as tf

def load_data(train_dir, test_dir, img_height=48, img_width=48, batch_size=32):
    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_data.class_names  # ✅ capture before map/prefetch
    print("Train and test data loaded successfully.")
    return train_data, test_data, class_names

