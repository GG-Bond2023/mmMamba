import matplotlib.pyplot as plt
import numpy as np
import cv2
from RadIal.ADCProcessing.DBReader.DBReader import SyncReader

# Open the dataset

# Load the camera calibration parameters
calib = np.load('/home/root/Documents/FFTRadNet/RadIal/ADCProcessing/DBReader/examples/camera_calib.npy',allow_pickle=True).item()


# Display the camera image
image = cv2.imread('/home/root/Documents/Dataset/RADIal/camera/image_000400.jpg')

plt.imshow(image)


# Get the laser point cloud
pts = np.load('/home/root/Documents/Dataset/RADIal/laser_PCL/pcl_000400.npy')
plt.plot(-pts[:,1],pts[:,0],'.')
plt.xlim(-15,15)
plt.ylim(0,100)
plt.grid()
print('Number of points:',pts.shape[0])
print('Parameters per point:',pts.shape[1])


def ConpensateLayerAngle(pcl, index, sensor_height):
    offset = 0
    if (index % 2 == 0):
        offset = np.deg2rad(.6)

    x = pcl[:, 4] * np.cos(pcl[:, 5] + offset) * np.cos(pcl[:, 6])
    y = pcl[:, 4] * np.cos(pcl[:, 5] + offset) * np.sin(pcl[:, 6])
    z = pcl[:, 4] * np.sin(pcl[:, 5] + offset) + sensor_height

    pcl[:, 0] = x
    pcl[:, 1] = y
    pcl[:, 2] = z

    return pcl


# %%
# Now, we can project 3D points into the camera space

pts = ConpensateLayerAngle(pts, 1, 0.42)[:, :3]
pts[:, [0, 1, 2]] = pts[:, [1, 0, 2]]  # Swap the order
pts[:, 0] *= -1  # Left is positive

imgpts, _ = cv2.projectPoints(np.array(pts),
                              calib['extrinsic']['rotation_vector'],
                              calib['extrinsic']['translation_vector'],
                              calib['intrinsic']['camera_matrix'],
                              calib['intrinsic']['distortion_coefficients'])

imgpts = imgpts.squeeze(1).astype('int')

# Keep only points inside the image size
idx = \
np.where((imgpts[:, 0] >= 0) & (imgpts[:, 0] < image.shape[1]) & (imgpts[:, 1] >= 0) & (imgpts[:, 1] < image.shape[0]))[
    0]

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.plot(imgpts[idx, 0], imgpts[idx, 1], 'r.')

plt.show()
