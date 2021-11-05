# Quantitative analysis for performance analysis
# @author Bereket Kebede, modified from @Dr Cong Van


import numpy as np
import math
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import sys

# The functions written are based on Luhong Jin 2020
def get_errors(gt,pr, data_range = None):

    # Width and Height of input picture
    W = 256
    H = 256

    # Standard deviation of the ground truth image
    Sd_gt = gt.std()

    # Normal root mean square error
    nrmse = np.linalg.norm( ((gt - pr) ** 2) / (W*H) )/ Sd_gt

    # Calculate PSNR
    psnr = 20 * math.log10(255 / nrmse * Sd_gt)

    def ssim(gt, pr, data_range = None):
        ssim = structural_similarity(gt, pr, win_size=None, gradient=False, data_range=None, multichannel=True)
        return ssim

    ssim = ssim(gt,pr)
    metrics = [round(nrmse,7), round(ssim,7), round(psnr,7)]
    return metrics

np.set_printoptions(threshold=sys.maxsize)


# control variables
# pred = plt.imread('D:/Bereket/DeepLearning/Analysis/Control variable/Sample_755.tif')
# ground = plt.imread('D:/Bereket/DeepLearning/Analysis/Control variable/Sample_755(2).tif')

# Test_1 - Luhong 2020 Trained, before and after restoration
# Test_3 - CIRL Trained
# Test 4 - CIRL Trained, between ground and predicted
# Test 5 - Luhong 2020 Trained, between ground and predicted

ground = plt.imread('D:/Bereket/DeepLearning/Testing_codes/UNet/Reproduction/Test_5/Sample_1.tif')
pred = plt.imread('D:/Bereket/DeepLearning/Testing_codes/UNet/Reproduction/Test_5/Sample_1_pred.tif')

pre_pred = pred[:,:, 0]
# pre_pred = pred # use for control variable

#print(pre_pred.shape)

print(pred.shape)
print(ground.shape)

results = get_errors(ground,pre_pred)

print("===========================================================")
print("   Comparing ground truth and prediction quantitatively    ")
print("===========================================================")
print("    NRMSE: |  SSIM: |  PSNR: ")
print(results)
print("===========================================================")

#print(np.amax(ground))

