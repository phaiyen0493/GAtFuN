import torch
import numpy as np
import hashlib

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model
    
def coco2humaneva_convert(x):
    y = np.zeros((x.shape[0], 15, x.shape[2]))
    y[:,0,:] = (x[:,11,:] + x[:,12,:])*0.5
    y[:,1,:] = (x[:,5,:] + x[:,6,:])*0.5
    y[:,2,:] = x[:,5,:]
    y[:,3,:] = x[:,7,:]
    y[:,4,:] = x[:,9,:]
    y[:,5,:] = x[:,6,:]
    y[:,6,:] = x[:,8,:]
    y[:,7,:] = x[:,10,:]
    y[:,8,:] = x[:,11,:]
    y[:,9,:] = x[:,13,:]
    y[:,10,:] = x[:,15,:]
    y[:,11,:] = x[:,12,:]
    y[:,12,:] = x[:,14,:]
    y[:,13,:] = x[:,16,:]
    y[:,14,:] = x[:,0,:]
        
    return y
    
def coco2humaneva(keypoints):
    keypoints = dict(keypoints)
    keypoints['metadata'] = np.array([{'layout_name': 'humaneva15', 'num_joints': 15, 'keypoints_symmetry': [[2,3,4,8,9,10], [5,6,7,11,12,13]]}], dtype=object)
    
    contents_2d = keypoints['positions_2d'].item()
    for subject in contents_2d.keys():
        for action in contents_2d[subject].keys():
            for cam in range(len(contents_2d[subject][action])):
                adjusted_2d = coco2humaneva_convert(contents_2d[subject][action][cam])
                keypoints['positions_2d'].item()[subject][action][cam] = adjusted_2d 
    return keypoints
    
    
def human36m2humaneva_convert(x):
    y = np.zeros((x.shape[0], 15, x.shape[2]))
    y[:,0,:] = x[:,0,:]
    y[:,1,:] = x[:,8,:]
    y[:,2,:] = x[:,11,:]
    y[:,3,:] = x[:,12,:]
    y[:,4,:] = x[:,13,:]
    y[:,5,:] = x[:,14,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = x[:,16,:]
    y[:,8,:] = x[:,4,:]
    y[:,9,:] = x[:,5,:]
    y[:,10,:] = x[:,6,:]
    y[:,11,:] = x[:,1,:]
    y[:,12,:] = x[:,2,:]
    y[:,13,:] = x[:,3,:]
    y[:,14,:] = x[:,9,:]
        
    return y
    
def human36m2humaneva(keypoints):
    keypoints = dict(keypoints)
    keypoints['metadata'] = np.array([{'layout_name': 'humaneva15', 'num_joints': 15, 'keypoints_symmetry': [[2,3,4,8,9,10], [5,6,7,11,12,13]]}], dtype=object)
    
    contents_2d = keypoints['positions_2d'].item()
    for subject in contents_2d.keys():
        for action in contents_2d[subject].keys():
            for cam in range(len(contents_2d[subject][action])):
                adjusted_2d = human36m2humaneva_convert(contents_2d[subject][action][cam])
                keypoints['positions_2d'].item()[subject][action][cam] = adjusted_2d 
    return keypoints
