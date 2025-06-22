import re

from glob import glob
import os, shutil
import random

class RadialDatasetSplit():
    # 如果不存在数据集索引文件train_idx、val_idx、test_idx，则创建
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
        print("开始拆分数据集")
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

        os.makedirs(img_train, exist_ok=True)  # 创建文件夹
        os.makedirs(img_val, exist_ok=True)  # 创建文件夹
        os.makedirs(img_test, exist_ok=True)  # 创建文件夹
        os.makedirs(seg_train, exist_ok=True)  # 创建文件夹
        os.makedirs(seg_val, exist_ok=True)  # 创建文件夹
        os.makedirs(seg_test, exist_ok=True)  # 创建文件夹
        # RADIal数据集里的ADC_Data文件数目要比radar_freespace文件数目多，无法做到一一对应，因此读取radar_freespace里的文件作为数据对。
        segmentations = [os.path.basename(f) for f in
                         glob(os.path.join(self.seg_map_path, '*' + self.seg_map_suffix))]  ## 获取所有样本文件
        if self.random:
            random.shuffle(segmentations)

        if self.img_suffix == '.npy':  # 处理adc_data
            samples = [s.replace('freespace', 'adc').replace(self.seg_map_suffix, self.img_suffix) for s in
                       segmentations]
        else:  # 处理其他数据
            assert NotImplementedError

        total = len(segmentations)
        # 计算各子集样本数量
        assert sum(self.split_ratio) == 1
        train_size = int(total * self.split_ratio[0])
        val_size = int(total * self.split_ratio[1])
        test_size = total - train_size - val_size
        # 划分数据集
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]

        train_segs = segmentations[:train_size]
        val_segs = segmentations[train_size:train_size + val_size]
        test_segs = segmentations[train_size + val_size:]

        # 为数据集创建软连接
        for s in train_samples:
            self.create_symlink(os.path.join('../', s), img_train, s)  # 为数据集创建软连接
        for s in val_samples:
            self.create_symlink(os.path.join('../', s), img_val, s)  # 为数据集创建软连接
        for s in test_samples:
            self.create_symlink(os.path.join('../', s), img_test, s)  # 为数据集创建软连接

        for s in train_segs:
            self.create_symlink(os.path.join('../', s), seg_train, s)  # 为数据集创建软连接
        for s in val_segs:
            self.create_symlink(os.path.join('../', s), seg_val, s)  # 为数据集创建软连接
        for s in test_segs:
            self.create_symlink(os.path.join('../', s), seg_test, s)  # 为数据集创建软连接

        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[0]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in train_samples]))
        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[1]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in val_samples]))
        # with open(os.path.join(self.data_split_info_dir, self.data_split_info[2]), 'w') as f:
        #     f.write('\n'.join([self.extract_digits(s) for s in test_samples]))
        print("完成数据集拆分作业。。。")


    def extract_digits(self, s):  # 提取字符串s里的数字部分
        match = re.search(r'\d+', s)
        return match.group(0) if match else s

    def check_files_exist(self):
        """检查指定文件夹下是否存在文件"""
        # 定义需要检查的文件名列表
        file_status = True
        # 检查每个文件是否存在
        for file in self.data_split_info:
            file_path = os.path.join(self.data_split_info_dir, file)
            exists = os.path.exists(file_path)
            file_status = exists and file_status
        return file_status

    def create_symlink(self, src_file, target_dir, link_name):  # 为数据集创建软连接
        os.makedirs(target_dir, exist_ok=True)
        link_path = os.path.join(target_dir, link_name)
        try:
            os.symlink(src_file, link_path)
        except FileExistsError:
            print(f"错误：软连接已存在 - {link_path}")
        except OSError as e:
            print(f"错误：创建软连接失败 - {e}")
            # 在某些系统上，可能需要管理员权限或特定设置才能创建软连接
            print("提示：如果权限不足，尝试以管理员身份运行脚本")



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
        # regenerate_split_info_file = True,#是否重新生成数据集索引文件
        random = True,#是否为train/val/test随机分配数据集索引文件
    )
    RDS.run()