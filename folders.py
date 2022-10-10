
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import pandas as pd


class DIYFolder(data.Dataset):
    """DIY Class, passing raw_dataset as the main folder (for now)."""

    def __init__(self, root, index, transform, patch_num):
        # /home/gohjiayi/Desktop/iqa/raw_dataset/live/databaserelease2
        live_root = os.path.join(root, 'live', 'databaserelease2')
        # /home/gohjiayi/Desktop/iqa/raw_dataset/liveitw/ChallengeDB_release
        liveitw_root = os.path.join(root, 'liveitw', 'ChallengeDB_release')
        # /home/gohjiayi/Desktop/iqa/raw_dataset/va
        va_root = os.path.join(root, 'va')

        live_index = list(range(0, 29))
        liveitw_index = list(range(0, 1162))
        va_index = list(range(0, 391))

        # LIVE
        refpath = os.path.join(live_root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(live_root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(live_root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(live_root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(live_root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(live_root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        # Original
        # NOTE: This includes perfect reference images, with MOS of 100
        dmos = scipy.io.loadmat(os.path.join(live_root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(live_root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        refname.sort()
        live_sample = []

        for i in range(0, len(live_index)):
            train_sel = (refname[live_index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    live_sample.append((imgpath[item], 100-labels[0][item])) # To convert DMOS to MOS


        # LIVEitW
        imgpath = scipy.io.loadmat(os.path.join(liveitw_root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169] # NOTE: omits the 7 training images
        mos = scipy.io.loadmat(os.path.join(liveitw_root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        liveitw_sample = []
        for i, item in enumerate(liveitw_index):
            for aug in range(patch_num):
                liveitw_sample.append((os.path.join(liveitw_root, 'Images', imgpath[item][0][0]), labels[item]))

        # VA
        va_sample = []
        imgpathmos = pd.read_csv(os.path.join(va_root, 'va_clean.csv'))

        for i, item in enumerate(va_index):
            for aug in range(patch_num):
                va_sample.append((os.path.join(va_root, 'processed', imgpathmos.iloc[item].filename), imgpathmos.iloc[item].mos))

        # Merge all 3 databases together and populate as a sample item
        self.samples = live_sample + liveitw_sample + va_sample

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class VAFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        sample = []
        imgpathmos = pd.read_csv(os.path.join(root, 'va_clean.csv'))
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'processed', imgpathmos.iloc[item].filename), imgpathmos.iloc[item].mos))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        length = len(self.samples)
        return length


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        # Original
        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        # TEMPORARY : running with z-score
        # Copied this in remember to remove /home/gohjiayi/Desktop/iqa/raw_dataset/live/databaserelease2/live_clean_standardised.csv
        # file = pd.read_csv(os.path.join(root, 'live_clean_standardised_982.csv'))
        # labels = np.array([file['standardised_mos'].to_numpy().astype(np.float32)])
        # orgs = np.array([file['orgs'].to_numpy()])
        # refnames_all = list(file['refnames_all'].to_numpy())
        # refnames_all = np.array([refnames_all])

        # # Debugggg to check the len of np array
        # # print(len(labels))
        # # print(len(orgs[0]))
        # print(len(refnames_all[0]))
        # print("="*100)
        # # print(len(new_labels))
        # # print(len(new_orgs[0]))
        # print(len(new_refnames_all[0]))
        # print("."*100)
        # print((refnames_all[0][0]))
        # print((new_refnames_all[0][0]))
        # # print(refname)


        refname.sort()
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all) # Image of the same filename, across all distortion types

            # new_train_sel = (refname[index[i]] == new_refnames_all)
            # print(train_sel[0][0], len(train_sel[0]), type(train_sel[0][0]))
            # print(new_train_sel[0][0], len(new_train_sel[0]), type(new_train_sel[0][0]))
            train_sel = train_sel * ~orgs.astype(np.bool_)

            train_sel = np.where(train_sel == True) # Find indexes where bool=True / n=1

            train_sel = train_sel[1].tolist() # Get indexes where bool=True / n=1
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], 100-labels[0][item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target, path

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath,'.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        refname.sort(reverse=True)
        # refnames_all.sort()

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'labels_image.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'].split('/')[1])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'FLIVE', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Kadid10k(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.png.PNG')
        # txtpath = os.path.join(root, 'dmos.txt')
        # fh = open(txtpath, 'r')



        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row['dist_img'])
                refnames_all.append(row['ref_img'][1:3])

                mos = np.array(float(row['dmos'])).astype(np.float32)
                target.append(mos)




        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)


        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')