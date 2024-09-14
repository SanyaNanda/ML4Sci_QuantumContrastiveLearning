from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

class LabeledContrastiveDataset(Dataset):
    """
    Take a folder containing sub-folders of images, where the sub-folder name is the image class, and generate
    pairs of images with the same class label for contrastive learning. This procedure fixes the batch size since
    every batch contains all classes and every batch element is a unique pairing of each class. 
    """

    def __init__(self, folder, transforms=None):
        
        labels = os.listdir(folder)
        
        self.labels_to_imgs = dict() # Dict with labels as keys and file paths as values
        self.idx_to_imgs = [] # List with tuples of each filename and label (label, fname)
        self.transform = transforms
        
        

        for label in labels: # Populate 
            current_dir = os.path.join(folder, label)
            label_files = [os.path.join(current_dir, x) for x in os.listdir(current_dir)]
            self.labels_to_imgs[label] = label_files
            self.idx_to_imgs += [(f, label) for f in label_files]
    
    def __len__(self):
        return(len(self.idx_to_imgs))
    
    def __getitem__(self, idx):
        '''
        
        
        '''
        # Grab the index image as the anchor
        img, label =  self.idx_to_imgs[idx]
        img = plt.imread(img)
        label_tensor = []
        
        # Grab an image with the same class as the anchor if available
        similar_imgs = len(self.labels_to_imgs[label])
        if similar_imgs > 2:
            similar_imgs_idx = np.random.choice(range(similar_imgs))
            similar_img = self.labels_to_imgs[label][similar_imgs_idx]
            similar_img = plt.imread(similar_img)
        else:
            raise NotImplementedError
        
        
        if self.transform is not None:
            img = self.transform(img); similar_img = self.transform(similar_img)
        
        
        out_tensor_x1 = img[np.newaxis,...]
        out_tensor_x2 = similar_img[np.newaxis,...]
        label_tensor.append(int(label))
        
        # To form the batch, grab all the other classes 
        for l in ((set(self.labels_to_imgs.keys())) - set(label)):
            
            dissimilar_imgs = len(self.labels_to_imgs[l])
            dissimilar_img_idx = np.random.choice(range(dissimilar_imgs))
            dissimilar_img = self.labels_to_imgs[l][dissimilar_img_idx]
            dissimilar_img = plt.imread(dissimilar_img)
            #dissimilar_img = np.expand_dims(dissimilar_img, 0)
            
            dissimilar_img_idx2 = np.random.choice(range(dissimilar_imgs))
            while dissimilar_img_idx2 == dissimilar_img_idx:
                dissimilar_img_idx2 = np.random.choice(range(dissimilar_imgs))

            dissimilar_img2 = self.labels_to_imgs[l][dissimilar_img_idx2]
            dissimilar_img2 = plt.imread(dissimilar_img2)
            #dissimilar_img2 = np.expand_dims(dissimilar_img2, 0)
    
            if self.transform is not None:
                dissimilar_img = self.transform(dissimilar_img)
                dissimilar_img2 = self.transform(dissimilar_img2)
            
            dissimilar_img = dissimilar_img[np.newaxis,...]
            dissimilar_img2 = dissimilar_img2[np.newaxis,...]
            
            out_tensor_x1 = torch.cat([out_tensor_x1, dissimilar_img])
            out_tensor_x2 = torch.cat([out_tensor_x2, dissimilar_img2])
            
            label_tensor.append(int(l))
        
        out_dict = {"x1": out_tensor_x1, "x2": out_tensor_x2, "labels": torch.Tensor(label_tensor)}
        return(out_dict)

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch

class LabeledContrastiveDatasetQG():
    """
    Dataset class to load images from .npz files, convert them to PyTorch tensors, and return x1 and x2.
    """

    def __init__(self, file, transforms=None):
        #self.files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.npz')]
        self.file=file
        self.transform = transforms
        
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        """
        Load the npz file, convert x1 and x2 to PyTorch tensors, and return them.
        """
        # file_path = self.files[idx]
        # data = np.load(file_path)
        
        data = np.load(self.file, allow_pickle=True)
        pairs = data["pairs"]
        labels = data["labels"]
        
        pairs = pairs[:,:,:,:,3]
        pairs = pairs.reshape(-1, 2, 125, 125, 1)
        x1 = pairs[:,0]
        x2 = pairs[:,1,]

        def crop_center(img,cropx,cropy):
            x,y = img.shape[1:3]
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)    
            return img[:,startx:startx+cropx,starty:starty+cropy,:]
        
        x1 = crop_center(x1,40,40)
        x2 = crop_center(x2,40,40)

        # Reshape the input tensors to add the channel dimension
        # x1 = x1.reshape(-1, 2, 125, 125, 1)
        # x2 = x2.reshape(-1, 2, 125, 125, 1)
        
        # Apply transforms if any
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        # Convert numpy arrays to PyTorch tensors
        x1_tensor = torch.tensor(x1, dtype=torch.float32).permute(0, 3, 1, 2)
        x2_tensor = torch.tensor(x2, dtype=torch.float32).permute(0, 3, 1, 2)
        labels = torch.Tensor(labels)

        return {"x1": x1_tensor, "x2": x2_tensor, 'labels':labels}