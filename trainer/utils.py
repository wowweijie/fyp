import numpy as np

def action_mask(action: np.ndarray, upper_thres, lower_thres):
    """ filter weak action signals between upper_thres and lower_thres to zero

    Args:
        action (ndarray): _description_
        upper_thres (_type_): _description_
        lower_thres (_type_): _description_
    """
    action[(action > -lower_thres) & (action < upper_thres)] = 0 