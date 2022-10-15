"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import pickle
import uuid

import pandas as pd


def dump(obj, exp_path, name, txt=False):
    """
    Save object either as a pickle or text file
    :param obj: object to save
    :param str exp_path: path where to save
    :param str name: name of the saved file
    :param boolean txt: if True, save as a text file
    """
    if not txt:
        with open(os.path.join(exp_path, name + ".pkl"), "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(os.path.join(exp_path, name + ".txt"), "w") as f:
            f.write(str(obj))


def load(exp_path, name):
    """
    Load a pickle object
    :param str exp_path: path to the file
    :param str name: name of the file
    """
    with open(os.path.join(exp_path, name), "rb") as f:
        obj = pickle.load(f)
    return obj


def np_to_csv(array, save_path):
    """
    Convert np array to .csv

    :param np.ndarray array: the array to convert to csv
    :param str save_path: where to temporarily save the csv
    :return: output_path, the path to the csv file
    """
    id = str(uuid.uuid4())
    output_path = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output_path, header=False, index=False)

    return output_path
