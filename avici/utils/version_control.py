import subprocess
import sys
import argparse
from datetime import datetime


def get_version():
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode(sys.stdout.encoding) 
    return "_" + git_commit


def get_datetime():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H%M")
    return "_" + date_time


def str2bool(v):
    v = "".join([char for char in v if u"\xa0" not in char])  # avoid utf8 parsing error
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 'T', 't', 'Y', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'F', 'f', 'N', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
