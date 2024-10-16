# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lib.models.pose_resnet
import lib.models.pose_hrnet_cond
from lib.models.pose_hrnet_nonlocal import *
from lib.models.pose_hrnet_lastfuse import *
from lib.models.pose_hrnet_threestage import *
from lib.models.pose_hrnet_threestageD import *
from lib.models.pose_hrnet_moreadd import *
from lib.models.pose_hrnet_morecat import *
from lib.models.pose_uhrnet import UHRNet
