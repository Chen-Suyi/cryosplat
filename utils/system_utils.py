#
# This implementation is based on the original Gaussian Splatting
# codebase released by the GRAPHDECO research group (Inria, 2023).
#
# Significant modifications have been made to the core computational
# components for research purposes.
#
# The original software is distributed under the Gaussian-Splatting License:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# This repository preserves the same licensing terms.
#


from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
