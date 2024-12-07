# Made by Jim.Wang V1 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable




from .Hailuo05 import Hailuo05,ImgCombine,GetBaPrompt

NODE_CLASS_MAPPINGS = {
    "Canvas": Hailuo05,
    "CombineImage": ImgCombine,
    "GetBAPrompt":GetBaPrompt
}


print('\033[34mHailuo03 Assistant Nodes: \033[92mLoaded\033[0m')