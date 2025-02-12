# Copyright (c) BLINDED
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from src.utils import RANDOM_SEED as SEED
np.random.seed(SEED)

def split_evenly_size(x, chunk_size, axis=0):
    return np.array_split(x, np.ceil(x.shape[axis] / chunk_size), axis=axis)


def divide_array_in_chunks(x, size, axis=0):
    """ Divide an array into chunks
    Parameters
    ----------
    x: numpy array
        The array to divide
    size: int
        The size of the chunks
    axis: int
        The axis to divide
    Returns
    -------
    chunks: list
        A list of arrays
    Examples
    --------
    >>> x = np.arange(10)
    >>> chunks = divide_array_in_chunks(x, 3)
    >>> print(chunks)

    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
    """
    return np.array_split(x, np.ceil(x.shape[axis] / size), axis=axis)
def split_index_folds(idx: np.ndarray, n_folds:int=5, axis:int=0, shuffle:bool=True)-> dict:

    """ Split the index into n_folds
    Parameters
    ----------
    idx: numpy array
        The index to split
    n_folds: int
        The number of folds
    axis: int
        The axis to split
    Returns
    -------
    folds: dict
        A dictionary with the index as key and the fold number as value
    Examples
    --------
    >>> idx = np.arange(10)
    >>> folds = split_index_folds(idx, 5)
    >>> print(folds)
    {'0': 0, '1': 0, '2': 1, '3': 1, '4': 2, '5': 2, '6': 3, '7': 3, '8': 4, '9': 4}
    """

    if shuffle: np.random.shuffle(idx)
    # splits = divide_array_in_chunks(idx, idx.shape[axis] // n_folds +1 , axis=axis)
    splits = np.array_split(idx, n_folds, axis=axis)
    indices  = {str(id): k for k, i in enumerate(splits) for id in i}
    return indices