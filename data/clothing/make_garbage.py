import pandas as pd
import os 
import numpy as np
from numpy.random import multivariate_normal


path = 'eos_embeddings_2024_12_11_13_30_30_indexed'
elem = os.listdir(path)

if not os.path.exists('gaussian_noise_indexed'):
    os.mkdir('gaussian_noise_indexed')
else:
    raise

if not os.path.exists('multivariate_noise_indexed'):
    os.mkdir('multivariate_noise_indexed')
else:
    raise

embs = []

for e in elem:
    embs.append(np.load(os.path.join(path, e)).squeeze())

embs = np.array(embs)
mean = embs.mean(axis=0)
cov = np.cov(embs.T) 
multivariate_noise = multivariate_normal(mean, cov, size=len(embs)) 
gaussian_noise = np.random.normal(loc=0.0, scale=1.0, size=embs.shape)

for i, e in enumerate(elem):
    np.save(os.path.join('multivariate_noise_indexed', e), multivariate_noise[i])
    np.save(os.path.join('gaussian_noise_indexed', e), gaussian_noise[i])