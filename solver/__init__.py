# adapted from [https://github.com/michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)

# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .build import make_optimizer, make_optimizer_with_center
from .lr_scheduler import WarmupMultiStepLR