# Using t-SNE to Visualize the CNN Features with PyTorch

### Note:

This script only contains a 2D t-SNE visualization

The source code located in ```code/t_sne.py```

## How to do this

### A. Generate and Save CNN Feature

In the first setp, you should generate CNN feature and save it to your disc.

- Import some modules

    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from dataloader import Data
    import tqdm
    import os
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    ```

    

- Define a feature extracter

    ```python
    class FeatureExtracter(nn.Module):
        """
        Extract the last fc layer features of the FusionBiGradNet
        In this section, we run the model without the classifier
        and to save the output features of the network
    """
        def __init__(self, model):
        	super(FeatureExtracter, self).__init__()
        	if isinstance(model, torch.nn.DataParallel):
            	model = model.module
       		 # remove the last FC layers
        	self.features = nn.Sequential(*list(model.children())[:-2])
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    ```



- Generate and save the extracted feature

    ```python
    def generate_feautres(model, data):
        model = FeatureExtracter(model)
        model.eval()
        features = []
        target = []
        for idx, batch in enumerate(data):
            image = batch[0].cuda()
            label = batch[1]
            # save the labels here
            target.append(label[:, np.newaxis])
            # perform forward process to obtain the features
            predictions = model(image)
            # convert the extracted features into cpu data
            predictions = predictions.data.cpu().numpy()
            features.append(predictions)
        features = np.concatenate(features, axis=0)
        target = np.concatenate(target, axis=0)
        save_path = "./extracted_features/"
        # save the extracted features with ".npy" format
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, "fusionbigradnet_features.npy"), features, allow_pickle=False)
        np.save(os.path.join(save_path, "fusionbigradnet_target.npy"), target, allow_pickle=False)
        # or, you can return the extracted features
        return features, target
    ```

    

- Then performing t-SNE visualizations

    ```python
    def t_sne():
        # load the extracted features
        features = np.load('./extracted_features/fusionbigradnet_features.npy').astype(np.float64)
        target = np.load('./extracted_features/fusionbigradnet_target.npy')
        # print("Generating CNN features...")
        # features, target = generate_feautres(model, test_data)
        # print("Generated features shape:{}, and target shape:{}".format(features.shape, target.shape))
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=10, init='pca', n_iter=5000, random_state=0)
        features_embedded = tsne.fit_transform(features)
        print("Embedded features shape:", features_embedded.shape)
        # normalize the embedded features
        f_min, f_max = features_embedded.min(0), features_embedded.max(0)
        features_embedded_normed = (features_embedded - f_min) / (f_max - f_min)
        print("Visualizing the t-SNE outputs...")
        t_sne_vis(features_embedded_normed, target)
        print("PROCESS DONE")
    
    
    def t_sne_vis(embedded_features, target):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=target, alpha=0.6)
        # also, you can label all these features with your label
        for i in range(embedded_features.shape[0]):
            plt.text(embedded_features[i, 0], embedded_features[i, 1], str(target[i][0] + 1), fontsize=5, verticalalignment='center',
                     horizontalalignment='center')
        plt.axis('off')
        # plt.show()
        plt.savefig('./t_sne.png', dpi=300)
        plt.close()
    ```

    

- At last, run this process in two steps

    ```python
    if __name__ == '__main__':
        # Stage 1: generate the CNN features
        generate_feautres(model=model, data=test_data)
        # Stage 2: t-SNE visualization
        t_sne()
    ```

    

## Visual Examples

![t_sne_example](http://ww1.sinaimg.cn/large/005CmS3Mgy1glul5k2nomj31hc140jxl.jpg)

