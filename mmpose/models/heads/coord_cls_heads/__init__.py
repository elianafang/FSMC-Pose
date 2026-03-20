# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .rtmw_head import RTMWHead
from .simcc_head import SimCCHead
from .rtmcc_head_SCSC import RTMCCHead_SCSC

__all__ = ['SimCCHead', 'RTMCCHead', 'RTMWHead', 'RTMCCHead_SCSC']
