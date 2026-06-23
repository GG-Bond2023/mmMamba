import re

from glob import glob
import os, shutil
import random

class RadialDatasetSplit():
    # Create dataset index files train_idx, val_idx, test_idx if they do not exist
    def __init__(self,
                 split_ratio,
                 # data_split_info_dir,
                 # data_split_info,
                 data_root,
                 img_path,
                 seg_map_path,
                 img_suffix,
                 seg_map_suffix,
                 # regenerate_split_info_file,
                 random,
                 ):
        self.split_ratio = split_ratio
        # self.data_split_info = data_split_info
        # self.data_split_info_dir = data_split_info_dir
        self.data_root = data_root
        self.img_path = os.path.join(data_root, img_path)
        self.seg_map_path = os.path.join(data_root, seg_map_path)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.random = random
        # self.regenerate_split_info_file = regenerate_split_info_file

    def run(self):
        print("Starting dataset split")
        # if os.path.exists(self.data_split_info_dir):
        #     shutil.rmtree(self.data_split_info_dir)

        img_train = os.path.join(self.img_path, 'train')
        if os.path.exists(img_train):
            shutil.rmtree(img_train)
        img_val = os.path.join(self.img_path, 'val')
        if os.path.exists(img_val):
            shutil.rmtree(img_val)
        img_test = os.path.join(self.img_path, 'test')
        if os.path.exists(img_test):
            shutil.rmtree(img_test)

        seg_train = os.path.join(self.seg_map_path, 'train')
        if os.path.exists(seg_train):
            shutil.rmtree(seg_train)
        seg_val = os.path.join(self.seg_map_path, 'val')
        if os.path.exists(seg_val):
            shutil.rmtree(seg_val)
        seg_test = os.path.join(self.seg_map_path, 'test')
        if os.path.exists(seg_test):
            shutil.rmtree(seg_test)

        os.makedirs(img_train, exist_ok=True)  # Create folder
        os.makedirs(img_val, exist_ok=True)  # Create folder
        os.makedirs(img_test, exist_ok=True)  # Create folder
        os.makedirs(seg_train, exist_ok=True)  # Create folder
        os.makedirs(seg_val, exist_ok=True)  # Create folder
        os.makedirs(seg_test, exist_ok=True)  # Create folder
        # The number of ADC_Data files in the RADIal dataset is greater than the number of radar_freespace files, so they cannot be directly paired. Read files from radar_freespace to form data pairs.
        # segmentations = [os.path.basename(f) for f in
        #                  glob(os.path.join(self.seg_map_path, '*' + self.seg_map_suffix))]  ## Get all sample files
        segmentations = []
        for f in glob(os.path.join(self.seg_map_path, '*' + self.seg_map_suffix)):# Get all sample files
            if(self.check_files_exist(self.img_path, f)): # Data that exists in radar_Freespace may not exist in ADC_Data, needs to be filtered out
                segmentations.append(os.path.basename(f) )

        if self.random:
            random.shuffle(segmentations)

        if self.img_suffix == '.npy':  # Process adc_data
            samples = [s.replace('freespace', 'adc').replace(self.seg_map_suffix, self.img_suffix) for s in
                       segmentations]
        else:  # Process other data
            assert NotImplementedError

        total = len(segmentations)
        # Calculate the number of samples for each subset
        assert sum(self.split_ratio) == 1
        train_size = int(total * self.split_ratio[0])
        val_size = int(total * self.split_ratio[1])
        test_size = total - train_size - val_size
        # Split the dataset
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]

        train_segs = segmentations[:train_size]
        val_segs = segmentations[train_size:train_size + val_size]
        test_segs = segmentations[train_size + val_size:]

        # Create symlink for dataset
        for s in train_samples:
            self.create_symlink(os.path.join('../', s), img_train, s)  # Create symlink for dataset
        for s in val_samples:
            self.create_symlink(os.path.join('../', s), img_val, s)  # Create symlink for dataset
        for s in test_samples:
            self.create_symlink(os.path.join('../', s), img_test, s)  # Create symlink for dataset

        for s in train_segs:
            self.create_symlink(os.path.join('../', s), seg_train, s)  # Create symlink for dataset
        for s in val_segs:
            self.create_symlink(os.path.join('../', s), seg_val, s)  # Create symlink for dataset
        for s in test_segs:
            self.create_symlink(os.path.join('../', s), seg_test, s)  # Create symlink for dataset

        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[0]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in train_samples]))
        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[1]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in val_samples]))
        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[2]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in test_samples]))
        print("Dataset split complete...")


    def extract_digits(self, s):  # Extract the numeric part of string s
        match = re.search(r'\d+', s)
        return match.group(0) if match else s

    def check_files_exist(self, img_path, segmentation):# Data that exists in radar_Freespace may not exist in ADC_Data, needs to be filtered out
        """Check if file exists in the specified folder"""
        # Define the list of filenames to check
        seg_filename = os.path.basename(segmentation)
        match = re.search(r'freespace_(\d+)\.png', seg_filename)
        number = match.group(1)
        adc_filename = f"adc_{number}.npy"
        adc_filepath = os.path.join(img_path, adc_filename)
        # Check if the file exists
        if os.path.exists(adc_filepath):
            return True
        else:
            return False

    def create_symlink(self, src_file, target_dir, link_name):  # Create symlink for dataset
        os.makedirs(target_dir, exist_ok=True)
        link_path = os.path.join(target_dir, link_name)
        try:
            os.symlink(src_file, link_path)
        except FileExistsError:
            print(f"Error: Symlink already exists - {link_path}")
        except OSError as e:
            print(f"Error: Failed to create symlink - {e}")
            # On some systems, administrator privileges or specific settings may be required to create symlinks
            print("Tip: If permissions are insufficient, try running the script as administrator")



if __name__ == '__main__':
    RDS = RadialDatasetSplit(
        split_ratio = [0.7,0.15,0.15],
        # data_split_info_dir = ,
        # data_split_info = ,
        data_root = '/home/root/Documents/RADIal_VMamba/segmentation/data/RADIal',
        img_path='ADC_Data',
        seg_map_path = 'radar_Freespace',
        img_suffix = '.npy',
        seg_map_suffix = '.png',
        # regenerate_split_info_file = True,# Whether to regenerate dataset index files
        random = True,# Whether to randomly assign dataset index files for train/val/test
    )
    RDS.run()