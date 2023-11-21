import os

import numpy as np

from gdcnet.data.utils import *
from gdcnet.model.gdcnet_vm import gdcnet_vm
from gdcnet.model.gdcnet_arch import SpatialTransformer
from gdcnet.model.metrics import calc_perf_metrics

def test_model(data_dir, model_dir, model_filename):
    # Model parameters
    model_params = model_filename.split('_')[1:]
    ndims = int(model_params[0][0])
    train_type = model_params[1].split('.')[0]
    if train_type == 'sup':
        network_arch = 'unet'
    elif train_type == 'semisup' or train_type == 'selfsup':
        network_arch = 'unet+stu'
    else:
        raise ValueError('train_type must be sup, semisup or selfsup')

    # Load the test data
    x_test, y_test = load_data_from_txt(data_dir, os.path.join(model_dir, 'test.txt'), ndims, training=False)

    # Build the model and load the weights
    inshape = x_test[0].shape[:-1]
    model = gdcnet_vm(inshape=inshape, arch=network_arch)
    model.load_weights(os.path.join(model_dir, model_filename))

    # Predict
    inputs = [np.expand_dims(x_test[..., 0], axis=-1), np.expand_dims(x_test[..., 1], axis=-1)]  # [EPId, T1w]
    if train_type == 'sup':
        vdm_pred = model.predict(inputs)
        # Unwrap the EPI images using the predicted VDM
        if ndims == 2:
            vdm_pred = np.concatenate([vdm_pred, np.zeros_like(vdm_pred)], axis=-1)
        elif ndims == 3:
            vdm_pred = np.concatenate([vdm_pred, np.zeros_like(vdm_pred), np.zeros_like(vdm_pred)], axis=-1)
        EPIdc = SpatialTransformer(interp_method='linear',
                                   indexing='ij',
                                   fill_value=None)([np.expand_dims(x_test[..., 0], axis=-1), vdm_pred])
        EPIdc = EPIdc.numpy()

    elif train_type == 'semisup' or train_type == 'selfsup':
        EPIdc, vdm_pred = model.predict(inputs)


    # Compute the performance metrics
    # Unroll the data to make it slice-wise if ndims == 3
    if ndims == 3:
        vdm_pred = np.concatenate(np.moveaxis(vdm_pred, 3, 1), axis=0)
        y_test = np.concatenate(np.moveaxis(y_test, 3, 1), axis=0)
        EPIdc = np.concatenate(np.moveaxis(EPIdc, 3, 1), axis=0)
        x_test = np.concatenate(np.moveaxis(x_test, 3, 1), axis=0)

    # VDM estimation performance: SSIM and PSNR
    vdm_pred = vdm_pred[...,0]
    vdm_gt = y_test[...,0]
    SSIM_vdm, PSNR_vdm = calc_perf_metrics(vdm_gt, vdm_pred)

    # EPI unwrapping performance: NMI, SSIM and PSNR
    EPIdc_gdcnet = EPIdc[...,0]
    EPIdc_bm = y_test[...,1]
    T1w = x_test[...,1]
    SSIM_dc, PSNR_dc, NMI_dc = calc_perf_metrics(EPIdc_bm, EPIdc_gdcnet, anat_ref=T1w, nmi_opt=True)

    # Print the results
    print('Model: {}'.format(model_filename))
    print('VDM estimation performance:')
    print('SSIM: {:.3f} +/- {:.3f}'.format(np.mean(SSIM_vdm), np.std(SSIM_vdm)))
    print('PSNR: {:.3f} +/- {:.3f}'.format(np.mean(PSNR_vdm), np.std(PSNR_vdm)))
    print('EPI unwrapping performance:')
    print('SSIM: {:.3f} +/- {:.3f}'.format(np.mean(SSIM_dc), np.std(SSIM_dc)))
    print('PSNR: {:.3f} +/- {:.3f}'.format(np.mean(PSNR_dc), np.std(PSNR_dc)))
    print('NMI: {:.3f} +/- {:.3f}'.format(np.mean(NMI_dc), np.std(NMI_dc)))




if __name__ == "__main__":
    data_dir = '../data/preprocessed/train_testID'
    models_dir = '../models/20231116'
    models_filenames = ['model_2D_sup.h5', 'model_2D_semisup.h5', 'model_2D_selfsup.h5', 'model_3D_sup.h5', 'model_3D_semisup.h5', 'model_3D_selfsup.h5']
    for model_i in models_filenames:
        test_model(data_dir, models_dir, model_i)