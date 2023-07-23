import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import numpy as np
from PIL import Image
from scipy import linalg
import pickle as pk

import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms as TF

from metric.inception import InceptionV3
from core.setting import config

cfg = config()

# prepare data
device = 'cpu'
global real_mean
global real_sigma

if os.path.isfile(cfg.fid_path):
    with open(cfg.fid_path, 'rb') as f:
        real_mean, real_sigma = pk.load(f)

def def_extractor(device, dim):
    """    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    } default is 2048 """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
    model = InceptionV3([block_idx]).to(device)
    return model

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            for transform in self.transforms:
                img = transform(img)
        return img

class ImageTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return self.tensor.shape[0]
    
    def __getitem__(self, i):
        return self.tensor[i]

def calculate_activation_statistics(datas, model, batch_size=50, dims=2048, data_type='tensor', device='cpu', num_workers=0):
    """Calculation of the statistics used by the FID.
    Params:
    -- datas       : List of image datas (path or tensor)
    -- model       : Instance of inception 
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(datas, model, batch_size, dims, data_type, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_activations(datas, model, batch_size=50, dims=2048, data_type='tensor', device='cpu', num_workers=0):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- datas       : List of image datas (path or tensor)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- data_type   : type of img data, have two option('path', 'tensor')
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(datas):
        batch_size = len(datas)
    
    if data_type == 'path':
        dataset = ImagePathDataset(datas, transforms=[TF.ToTensor(), TF.Resize([64,64])])
    elif data_type == 'tensor':
        dataset = ImageTensorDataset(datas)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(datas), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calc_mean_sigma_path(path, feat_extractor):
    path_list = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            path_list.append(os.path.join(root, file))
    mean, sigma = calculate_activation_statistics(path_list, feat_extractor, data_type='path', device=next(feat_extractor.parameters()).device)
    return mean, sigma

def calc_mean_sigma_tensor(tensor, feat_extractor):
    mean, sigma = calculate_activation_statistics(tensor, feat_extractor, data_type='tensor', device=next(feat_extractor.parameters()).device)
    return mean, sigma

def calculate_fid_path(fake_path, feat_extractor):
    global real_mean
    global real_sigma
        
    fake_mean, fake_sigma = calc_mean_sigma_path(fake_path, feat_extractor)
    fid = calculate_frechet_distance(real_mean, real_sigma, fake_mean, fake_sigma)
    return fid

def calculate_fid_tensor(generator, input_shape, feat_extractor):
    global real_mean
    global real_sigma
    
    if generator.device is not next(feat_extractor.parameters()).device:
        generator = generator.to(next(feat_extractor.parameters()).device)

    g_out_list = []
    with torch.no_grad():
        for i in range(50000):
            z = torch.randn(input_shape, device=generator.device)
            g_out = generator(z)
            g_out_list.append(g_out)
            g_out_tensor = torch.cat(g_out_list, dim=0)
    fake_mean, fake_sigma = calc_mean_sigma_tensor(g_out_tensor, feat_extractor)

    fid = calculate_frechet_distance(real_mean, real_sigma, fake_mean, fake_sigma)
    return fid

def calculate_fid_tensor_real_time(fid_path, g_out, feat_extractor):
    global real_mean
    global real_sigma
    
    fake_mean, fake_sigma = calc_mean_sigma_tensor(g_out, feat_extractor)

    fid = calculate_frechet_distance(real_mean, real_sigma, fake_mean, fake_sigma)
    return fid

if __name__ == '__main__':
    None
