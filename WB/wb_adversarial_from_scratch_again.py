import pandas as pd
import numpy as np
import random

from bisect import bisect_left
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StandardDataset
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

