import numpy as np


def dot_sim(rv_i:np.ndarray, cv_w:np.ndarray, dot_len= 64, cv_bits=4, tia_noise_sigma=3.0, out_bits=8):

    if abs(np.max(cv_w)) > 2**(cv_bits-):
        raise ValueError("weights doesn't have correct data type.")
    if rv_i.shape[-1] != dot_len:
        raise ValueError("input columns neeed to be {dot_len}")
    if cv_w.shape[0] != dot_len:
        raise ValueError("weight rows neeed to be {dot_len}")
    
    out_mat = np.matmul(rv_i.astype(np.int), cv_w.astype(np.int))
    out_mat += np.random.normal(loc=128, scale=tia_noise_sigma, size=out_mat.shape)
    out_interm = np.round(np.clip(out_mat, 0, 255))

    results = out_interm.astype(np.uint8) >> (8-out_bits)


    return results

