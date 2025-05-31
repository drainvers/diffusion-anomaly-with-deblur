import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
from scipy import ndimage
from PIL import Image
from textwrap import wrap

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class ChexpertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory,
        class_cond=True,
        test_flag=False,
        sample_n=None,
        transform=None,
        data_filter=None,
        frontal_mode="both",
        unique_patients_only=False
    ):
        if not root_directory:
            raise ValueError("unspecified data directory")
        self.directory = os.path.expanduser(root_directory)
        self.test_flag = test_flag
        self.class_cond = class_cond
        self.annotations_path = os.path.join(self.directory,
                                             "valid.csv"
                                             if self.test_flag else
                                             "train.csv")
        self.transform = transform

        # Filtering logic
        df_annotations = pd.read_csv(self.annotations_path)

        if data_filter == "frontal_only":
            df_annotations = df_annotations[(df_annotations["Frontal/Lateral"] == "Frontal")]
            if frontal_mode == "ap":
                df_annotations = df_annotations[(df_annotations["AP/PA"] == "AP")]
            elif frontal_mode == "pa":
                df_annotations = df_annotations[(df_annotations["AP/PA"] == "PA")]

        self.local_classes = None
        if class_cond:
            # Generate labels for healthy images and images with pleural effusions
            df_annotations.loc[(df_annotations["No Finding"] == 1), 'Label'] = 0 # Healthy
            df_annotations.loc[(df_annotations["Pleural Effusion"] == 1), 'Label'] = 1 # Diseased

            df_annotations_healthy = df_annotations[(df_annotations["Label"] == 0)]
            df_annotations_diseased = df_annotations[(df_annotations["Label"] == 1)]

            if unique_patients_only:
                df_annotations_healthy = self._clean_df(df_annotations_healthy)
                df_annotations_diseased = self._clean_df(df_annotations_diseased)

            # Sample from pleural effusion images to lower imbalance
            if sample_n:
                if len(df_annotations_healthy.index) > sample_n:
                    df_annotations_healthy = df_annotations_healthy.sample(n=sample_n, random_state=1911)
                if len(df_annotations_diseased.index) > sample_n:
                    df_annotations_diseased = df_annotations_diseased.sample(n=sample_n, random_state=1911)

            df_annotations = pd.concat([df_annotations_diseased, df_annotations_healthy])

            self.local_classes = df_annotations['Label'].to_list()
        
        df_annotations['Path'] = df_annotations['Path'].apply(
                                    lambda path: os.path.join(
                                        self.directory, 
                                        os.sep.join(path.split(os.sep)[1:]))).astype(str)
        self.local_images = df_annotations['Path'].to_list()

        if class_cond:
            assert len(self.local_images) == len(self.local_classes)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        basename, ext = os.path.splitext(os.path.basename(path))
        name = basename
        print(name)

        # Readds ability to read known image formats, taken from upstream guided diffusion repo
        if ext == '.npy':
            out_img = np.load(path)
        else:
            out_img = np.asarray(Image.open(path).convert('L')) # Use Pillow for TIF support
            out_img = np.expand_dims(out_img, axis=2) # Changes image shape to (H, W, 1)
        
        out_img = visualize(out_img).astype(np.float32)
        out_img = np.transpose(out_img, [2, 0, 1]) # HWC -> CHW

        out_dict = {}
        if self.local_classes:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["name"] = name
        
        if self.transform:
            out_img = self.transform(out_img)

        return out_img, out_dict
    
    def summarize(self):
        print('n_images  :', len(self.local_images))
        if self.class_cond:
            print('n_labels  :', len(self.local_classes))
            print('n_healthy :', int(sum(self.local_classes)))
            print('n_diseased:', len(self.local_classes) - int(sum(self.local_classes)))
    
    def _clean_df(self, input_df):
        # Do NOT use first(), it will return the first non-NaN value, use nth to get values as-is
        input_df.loc[:, 'Patient ID'] = input_df['Path'].map(lambda x: os.path.basename(x).split('_')[0])
        return input_df.groupby('Patient ID').nth(0).reset_index(drop=True)

    def __len__(self):
        return len(self.local_images)

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number=filedict['t1'].split('/')[4]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(
                number) + '-label.nii.gz'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database)

def preview_dataset(ds, nrow, ncol):
    fig = plt.subplots(figsize=(nrow * 2.56, ncol * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)

    for row in range(nrow):
        for col in range(ncol):
            im, out_dict = ds[ncol * row + col]
            ax = plt.subplot(gs[row, col])
            ax.imshow(np.transpose(im, [1, 2, 0]), cmap='gray', aspect='auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(top=False, right=False, bottom=False, left=False, pad=0)
            ax.set(frame_on=False)
            if row == 0:
                ax.set_xlabel('\n'.join(wrap(out_dict['name'], 20)))
                ax.xaxis.set_label_position('top')
            if col == 0:
                ax.set_ylabel('\n'.join(wrap(out_dict['name'], 20)), wrap=True)
                ax.yaxis.set_label_position('left')
    
    plt.tight_layout(pad=1)
    plt.savefig('./chexpert_sample.png', dpi=100)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    train_ds = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=False, data_filter="frontal_only")
    test_ds = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=True, data_filter="frontal_only")
    # preview_dataset(train_ds, 3, 3)
    train_ds.summarize()
    test_ds.summarize()