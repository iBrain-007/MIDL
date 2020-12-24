import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
import numpy as np
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

"""as 
    To visualize the raw images with t-SNE
"""


def load_dataset(filepath):
    """
    The raw images must be converted to numpy arrays
    :param filepath: the raw image path
    :return:
    """
    img_names = os.listdir(filepath)
    # we want to resize the image to (224,224), so the data size is: (len_imgs, 224 * 224)
    data = np.zeros((len(img_names), 50176))
    # label = np.zeros((len(img_names), 1))
    label = []
    for i in range(len(img_names)):
        img_path = os.path.join(filepath, img_names[i])
        # save the label index into np.array
        idx = int(img_names[i][-5:-4]) - 1
        idx = np.asarray([idx])
        if idx == 0 or idx == 1:
            label.append(idx[:, np.newaxis])
            img = cv2.imread(img_path, flags=-1)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 50176)
            data[i] = img
    n_samples, n_features = data.shape
    label = np.concatenate(label, axis=0)
    data = data.astype(np.float64)
    return data, label


def plot_2d(data, label):
    # normalize the data
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(data[:, 0], data[:, 1], c=label, alpha=0.6)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i][0] + 1), fontsize=5, verticalalignment='center',
                 horizontalalignment='center')
    plt.axis('off')
    plt.savefig('./extracted_features/dataset_tsne(0-1).png', dpi=300)
    plt.close()


if __name__ == '__main__':
    # features = np.load('./extracted_features/fusionbigradnet_features.npy').astype(np.float64)
    # target = np.load('./extracted_features/fusionbigradnet_target.npy')
    # print(target.shape)
    path = "/path/to/raw/images/"
    print("loading dataset...")
    data, label = load_dataset(path)
    print(data.shape, label.shape)
    print("performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
    embeded_data = tsne.fit_transform(data)
    print("plotting embedded features...")
    plot_2d(embeded_data, label)

    # load_dataset(path)
