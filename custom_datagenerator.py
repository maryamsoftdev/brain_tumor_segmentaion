import os
import numpy as np

def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:
        if image_name.endswith('.npy'):
            try:
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    return np.array(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    X_batches, Y_batches = [], []
    batch_start = 0
    while batch_start < L:
        batch_end = min(batch_start + batch_size, L)
        X_batch = load_img(img_dir, img_list[batch_start:batch_end])
        Y_batch = load_img(mask_dir, mask_list[batch_start:batch_end])
        if len(X_batch) == 0 or len(Y_batch) == 0:  # Skip empty batches
            break
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
        batch_start += batch_size

    return X_batches, Y_batches

# Test function to verify the generator
def test_imageLoader():
    import random
    import matplotlib.pyplot as plt

    train_img_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\images"
    train_mask_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\masks"
    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    batch_size = 2

    X_batches, Y_batches = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

    # Visualize some examples from the generator
    img_num = random.randint(0, len(X_batches) - 1)
    test_img = X_batches[img_num]
    test_mask = Y_batches[img_num]
    test_mask = np.argmax(test_mask, axis=3)

    n_slice = random.randint(0, test_mask.shape[2] - 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.imshow(test_img[n_slice, :, :, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(test_img[n_slice, :, :, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(test_img[n_slice, :, :, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:, :, n_slice])
    plt.title('Mask')
    plt.show()

if __name__ == "__main__":
    test_imageLoader()
