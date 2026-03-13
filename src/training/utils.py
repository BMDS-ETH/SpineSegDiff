
import os 
import glob 
import torch
import logging
import torch.nn as nn
from pathlib import Path

def delete_old_weights(dirpath, filename_regex=""):
    """
    Delete the last model with the symbol in the model_dir
    Parameters:
    ------------
    model_dir: str
        The directory where the model is saved
    symbol: str
        The symbol to delete
    Returns:
    ---------
    None
    """
    try:
        old_model = glob.glob(f"{dirpath}/{filename_regex}*.pt")
        if len(old_model) != 0:
            os.remove(old_model[0])
    except Exception as e:
        logging.error(e)

def save_new_model_weights(model: nn.Module, save_path: [str, Path], delete_regex=None):
    """
    Save the model weights in the save_path. If delete_symbol is not None,
    the last model with the delete_symbol will be deleted.
    Parameters:
    -----------
    model: nn.Module
        The model to save
    save_path: str
        The path to save the model weights
    delete_symbol: str
        The symbol to delete the last model with this symbol
    Returns:
    --------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if delete_regex is not None:
        delete_old_weights(save_path, delete_regex)
    try:
        torch.save(model.state_dict(), save_path.as_posix())
        logging.info(f"model is saved in {save_path}")
    except Exception as e:
        logging.error(e)