# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:34:50 2019

@author: qianliang

train data from https://nlp.stanford.edu/projects/nmt/
pytorch version of "Effective Approaches to Attention-based Neural Machine Translation"
this  file mainly focus on local attention
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata

import re
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "global"
#model = "local"








