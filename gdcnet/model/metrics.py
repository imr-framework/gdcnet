import math
import numpy as np
import skimage.metrics as skm

def calc_perf_metrics(vol_ref, vol_pred, anat_ref=None, nmi_opt=False):
    '''
    Parameters
    ----------
    vol_ref : numpy array
        Reference volume.
    vol_pred : numpy array
        Predicted volume.
    anat_ref : numpy array, optional
        Reference anatomical volume.
    nmi_opt : bool, optional
        If True, calculate NMI.
    Returns
    -------
    ssim : numpy array
        SSIM value for each slice.
    psnr : numpy array
        PSNR value for each slice.
    nmi : numpy array, optional
        NMI value for each slice.
    '''

    nslices = vol_ref.shape[0]
    ssim, psnr, nmi = [], [], []

    if nmi_opt == False:
        vol_ref[np.where(np.abs(vol_ref) <= 1e-3)] = 0.0
        vol_pred[np.where(np.abs(vol_ref) == 0.0)] = 0.0 # Mask the predicted VDM to compare with the reference VDM

    for i in range(nslices):

        singal_dev = (vol_ref[i].max() - vol_ref[i].min())

        if singal_dev <= 0: # Check that there is signal in the slice
            pass
        else:
            # SSIM and PSNR
            vdm_range = np.max([np.max(vol_ref[i]), np.max(vol_pred[i])]) - np.min(
                [np.min(vol_ref[i]), np.min(vol_pred[i])])
            psnr.append(skm.peak_signal_noise_ratio(vol_ref[i], vol_pred[i], data_range=vdm_range))
            ssim.append(skm.structural_similarity(vol_ref[i], vol_pred[i], data_range=vdm_range))

        # NMI
        if nmi_opt:
            nmi.append(normalized_mutual_information(anat_ref[i].ravel(), vol_pred[i].ravel()))

    if nmi_opt:
        return ssim, psnr, nmi
    else:
        return ssim, psnr

def normalized_mutual_information(im1, im2, bins=100):
    '''
    Parameters
    ----------
    im1 : numpy array
        Reference image.
    im2 : numpy array
        Predicted image.
    bins : int
        Number of bins for histogram.
    Returns
    -------
    nmi : float
        Normalized mutual information.
    '''
    # from scipy.stats import entropy
    # H0 = entropy(np.sum(hgramjoint, axis=0))
    # H1 = entropy(np.sum(hgramjoint, axis=1))
    # H01 = entropy(np.reshape(hgramjoint, -1))

    hgram1 = np.histogram(im1, bins=bins)
    hgram2 = np.histogram(im2, bins=bins)  # histogram
    hgramjoint, _ = np.histogramdd([im1, im2], bins=bins, density=True)  # joint histogram

    p1 = hgram1[0] / np.sum(hgram1[0])
    nzs = p1 > 0
    H1 = -np.sum(p1[nzs] * np.log(p1[nzs]))

    p2 = hgram2[0] / np.sum(hgram2[0])
    nzs = p2 > 0
    H2 = -np.sum(p2[nzs] * np.log(p2[nzs]))

    pjoint = hgramjoint / np.sum(hgramjoint)
    p1_joint = np.sum(pjoint, axis=1)
    p2_joint = np.sum(pjoint, axis=0)
    p1_p2_joint = p1_joint[:, None] * p2_joint[None, :]
    nzs = pjoint > 0
    Hjoint = np.sum(pjoint[nzs] * np.log(pjoint[nzs] / p1_p2_joint[nzs]))

    return 2 * Hjoint / (H1 + H2)  # (H0 + H1) / H01 - 1.00 #
