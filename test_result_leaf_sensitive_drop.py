import sys
import os
import json
from collections import deque

import numpy as np
import pickle

from scipy.optimize import minimize
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE

from tqdm import tqdm
import pandas as pd

k_folds = 3

#########
root_dir = ""
data_name = "LEAF"
#########
print("sesitivities 1")

m_metric = np.array([[1.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.25, 1.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.25, 0.25, 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.25, 0.25, 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 1.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 1.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 1.  , 0.25, 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 1.  , 0.25, 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.  , 0.25, 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.  , 0.25, 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.  , 0.25],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.  ]])

# print(m_metric.size)

# print("sesitivities 3")
# m_metric = np.eye(39, dtype=float)

# print("sesitivities 2")
# m_metric = np.array([
#     [1.00, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
#     [0.25, 1.00, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.50],
#     [0.25, 0.25, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25, 0.25],
#     [0.25, 0.25, 0.50, 1.00, 0.50, 0.50, 0.50, 0.50, 0.25, 0.25],
#     [0.25, 0.25, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50, 0.25, 0.25],
#     [0.25, 0.25, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.25, 0.25],
#     [0.25, 0.25, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.25, 0.25],
#     [0.25, 0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.25, 0.25],
#     [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.00, 0.25],
#     [0.25, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.00]
# ])

# print("sesitivities 3")
# m_metric = np.array([
#     [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
#     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
# ])

# >>> PRECISE PREDICTIONS >>>

def evaluate(output, labels, n_class, save_dir, fold, test_type):        
    all_p_star_L1 = np.zeros([len(labels), n_class])
    all_p_star_KLD = np.zeros([len(labels), n_class])
    eval_results = []

    # >>> Squared Euclidean distance >>>

    # pred_mean shape ([10000, 10])
    # all_p_star_SED = pred_mean
    all_p_star_SED = np.mean(output, axis=0)    

    # <<< Squared Euclidean distance <<<

    for i in range(len(labels)):
        probability_set = output[:,i,:] #(100,10)

        # >>> L1_distance >>>

        def L1_distance(p_star_L1, probability_set):
            return np.sum(np.abs(probability_set - p_star_L1))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

        result = minimize(L1_distance, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)    
        p_star_L1 = result.x

        # all_p_star_L1 shape [10000, 10]
        all_p_star_L1[i] = p_star_L1

        # <<< L1_distance <<<

        # >>> KL_divergence >>>

        def KL_divergence(p_star_KLD, probability_set):
            epsilon = 1e-10
            p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)
            probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
            return np.sum(p_star_KLD * np.log(p_star_KLD/probability_set))
            # return np.sum(probability_set * np.log(probability_set/p_star_KLD))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                {'type': 'ineq', 'fun': lambda x: x}]
        
        result = minimize(KL_divergence, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)
        p_star_KLD = result.x
        epsilon = 1e-10
        p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)

        # all_p_star_KLD shape [10000, 10]
        all_p_star_KLD[i] = p_star_KLD

        # <<< KL_divergence <<<     

    # >>> Save test outputs >>>

    all_p_star_SED_sensitive = np.matmul(all_p_star_SED, m_metric)
    predictions_SED = np.argmax(all_p_star_SED_sensitive, axis=1)
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_SED = (predictions_SED == labels).mean() * 100
    
    all_p_star_L1_sensitive = np.matmul(all_p_star_L1, m_metric)
    predictions_L1 = np.argmax(all_p_star_L1_sensitive, axis=1)
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_L1 = (predictions_L1 == labels).mean() * 100

    all_p_star_KLD_sensitive = np.matmul(all_p_star_KLD, m_metric)
    predictions_KLD = np.argmax(all_p_star_KLD_sensitive, axis=1)    
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_KLD = (predictions_KLD == labels).mean() * 100


    np.save(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy', all_p_star_SED)
    np.save(f'{save_dir}predictions_SED_sensitive_{test_type}_fold_{fold}.npy', predictions_SED)

    np.save(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy', all_p_star_L1)
    np.save(f'{save_dir}predictions_L1_sensitive_{test_type}_fold_{fold}.npy', predictions_L1)

    np.save(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy', all_p_star_KLD)
    np.save(f'{save_dir}predictions_KLD_sensitive_{test_type}_fold_{fold}.npy', predictions_KLD)

    # <<< Save test output <<<

    

    acc_reward_sensitive_SED = 0
    acc_reward_sensitive_L1D = 0
    acc_reward_sensitive_KLD = 0
    for idx in range(len(all_p_star_SED)):
        acc_reward_sensitive_SED += m_metric[predictions_SED[idx], labels[idx]]
        acc_reward_sensitive_L1D += m_metric[predictions_L1[idx], labels[idx]]
        acc_reward_sensitive_KLD += m_metric[predictions_KLD[idx], labels[idx]]
        
    eval_results.append(acc_reward_sensitive_SED)
    eval_results.append(acc_reward_sensitive_L1D)
    eval_results.append(acc_reward_sensitive_KLD)

    return eval_results

train_type = "clean"
data_name = "LEAF-C"
model_type = "Drop_out"

print(train_type)
save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/{model_type}/'
# out_dir = save_dir + "cala_no_reward/"
# print(out_dir)
# if not os.path.exists(out_dir):
#         print(f"Path does not exist. Creating: ")
#         os.makedirs(out_dir)
# print(root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/')

kfold_results_c = []
kfold_results_clean = []

for fold in range(k_folds):
    fold = 2
    print(f'FOLD {fold}')
    print('--------------------------------')

        
    # output shape ([100, 10000, 10])
    output_c = np.load(f'{save_dir}tensor_output_c_fold_{fold}.npy')
    labels_c = np.load(f'{save_dir}test_labels_c_fold_{fold}.npy')
    output_clean = np.load(f'{save_dir}tensor_output_clean_fold_{fold}.npy')
    labels_clean = np.load(f'{save_dir}test_labels_clean_fold_{fold}.npy')
    labels_onehot_c = np.load(f'{save_dir}test_labels_onehot_c_fold_{fold}.npy')
    labels_onehot_clean = np.load(f'{save_dir}test_labels_onehot_clean_fold_{fold}.npy')

    # with open('./iukm/cifar_test_data_class_to_idx.txt', 'r') as f:
    #     class_to_idx = json.load(f)
    # n_class = len(class_to_idx) # = 10

    n_class = 39
    
    kfold_results_c.append(evaluate(output_c, labels_c, n_class, save_dir, fold, 'c'))
    kfold_results_clean.append(evaluate(output_clean, labels_clean, n_class, save_dir, fold, 'clean'))
    break
# print(np.round(kfold_results_c, 2))
# print('-'*8)
# print(np.round(kfold_results_clean, 2))

sys.exit()

#clean - noise
# [[38.94 38.98 38.79]          [35.04 35.20 35.05]                 [43.78 44.05 44.02]
#  [34.14 34.06 34.26]          [34.36 34.62 34.35]                 [32.79 33.01 33.58]
#  [40.02 39.94 39.94]]         [42.00 42.32 42.30]                 [40.34 40.40 40.63]
# -------- clean - clean
# [[87.02 87.06 87.02]          [86.94 86.84 86.86]                 [86.84 86.77 86.80]
#  [86.81 86.89 86.8 ]          [86.68 86.65 86.71]                 [86.40 86.28 86.40]
#  [86.98 86.9  86.94]]         [86.96 86.86 86.89]                 [86.50 86.44 86.50]

# noise - noise
# [[80.3  80.28 80.26]          [80.06 79.98 80.08]                 [80.30 80.14 80.24]
#  [80.12 79.98 79.96]          [80.02 80.00 79.97]                 [78.79 78.81 78.79]
#  [79.92 79.85 79.81]]         [80.06 80.03 80.04]                 [80.05 80.10 80.05]
# -------- noise - clean
# [[78.7  78.49 78.52]          [80.36 80.23 80.10]                 [78.34 78.30 78.80]
#  [78.4  78.31 78.27]          [79.12 79.06 78.82]                 [78.55 78.48 78.33]
#  [78.8  78.67 78.68]]         [78.91 78.81 78.81]                 [79.77 79.68 79.48]

# kfold_results_c = np.array([[38.94, 38.98, 38.79],
#                             [34.14, 34.06, 34.26],
#                             [40.02, 39.94, 39.94]])

# kfold_results_clean = np.array([[87.02, 87.06, 87.02],
#                                 [86.81, 86.89, 86.80],
#                                 [86.98, 86.90, 86.94]])

# kfold_results_c_mean = np.round(np.mean(kfold_results_c, axis=0),2)
# kfold_results_c_std  = np.round(np.std(kfold_results_c, axis=0),2)

# kfold_results_clean_mean = np.round(np.mean(kfold_results_clean, axis=0),2)
# kfold_results_clean_std  = np.round(np.std(kfold_results_clean, axis=0),2)

# print(kfold_results_c_mean)
# print(kfold_results_c_std)

# print('-'*11)

# print(kfold_results_clean_mean)
# print(kfold_results_clean_std)

# sys.exit()

# <<< PRECISE PREDICTIONS <<<

# >>> CALIBRATION ERRORS >>>

# # [[SED_c L1_c KLD_c SED_clean L1_clean KLD_clean], [fold 2], [fold 3]]
# kfold_ece_class = []
# kfold_mce_class = []
# kfold_ece_confi = []
# kfold_mce_confi = []

# for fold in range(k_folds):
#     print(f'FOLD {fold}')
#     print('--------------------------------')

#     # [SED_c L1_c KLD_c SED_clean L1_clean KLD_clean]
#     ece_class = []
#     mce_class = []
#     ece_confi = []
#     mce_confi = []
    
#     # fold = 0
#     save_dir = './data/CIFAR-10-C/gaussian_noise_c_fold/'

#     if not os.path.exists(f'{save_dir}calibration_error.txt'):
#         print('File not exists.')
#         sys.exit()

#     all_p_star_SED_c =  np.load(f'{save_dir}all_p_star_SED_sensitive_c_fold_{fold}.npy')
#     all_p_star_L1_c =   np.load(f'{save_dir}all_p_star_L1_sensitive_c_fold_{fold}.npy')
#     all_p_star_KLD_c =  np.load(f'{save_dir}all_p_star_KLD_sensitive_c_fold_{fold}.npy')

#     predictions_SED_c = np.load(f'{save_dir}predictions_SED_sensitive_c_fold_{fold}.npy')
#     predictions_L1_c =  np.load(f'{save_dir}predictions_L1_sensitive_c_fold_{fold}.npy')
#     predictions_KLD_c = np.load(f'{save_dir}predictions_KLD_sensitive_c_fold_{fold}.npy')

#     all_p_star_SED_clean =  np.load(f'{save_dir}all_p_star_SED_sensitive_clean_fold_{fold}.npy')
#     all_p_star_L1_clean =   np.load(f'{save_dir}all_p_star_L1_sensitive_clean_fold_{fold}.npy')
#     all_p_star_KLD_clean =  np.load(f'{save_dir}all_p_star_KLD_sensitive_clean_fold_{fold}.npy')

#     predictions_SED_clean = np.load(f'{save_dir}predictions_SED_sensitive_clean_fold_{fold}.npy')
#     predictions_L1_clean =  np.load(f'{save_dir}predictions_L1_sensitive_clean_fold_{fold}.npy')
#     predictions_KLD_clean = np.load(f'{save_dir}predictions_KLD_sensitive_clean_fold_{fold}.npy')

#     labels_c = np.load(f'{save_dir}cifar_test_labels_c_fold_{fold}.npy')
#     labels_clean = np.load(f'{save_dir}cifar_test_labels_clean_fold_{fold}.npy')
#     labels_onehot_c = np.load(f'{save_dir}cifar_test_labels_onehot_c_fold_{fold}.npy')
#     labels_onehot_clean = np.load(f'{save_dir}cifar_test_labels_onehot_clean_fold_{fold}.npy')

#     print('acc_SED_c:', (predictions_SED_c == labels_c).mean() * 100)
#     print('acc_L1_c:',  (predictions_L1_c == labels_c).mean() * 100)
#     print('acc_KLD_c:', (predictions_KLD_c == labels_c).mean() * 100)

#     print('acc_SED_clean:', (predictions_SED_clean == labels_clean).mean() * 100)
#     print('acc_L1_clean:',  (predictions_L1_clean == labels_clean).mean() * 100)
#     print('acc_KLD_clean:', (predictions_KLD_clean == labels_clean).mean() * 100)

#     # print('binary_ECE_SED:', binary_ECE(labels_onehot, all_p_star_SED, bins=30))
#     # print('binary_ECE_L1:', binary_ECE(labels_onehot, all_p_star_L1, bins=30))
#     # print('binary_ECE_KLD:', binary_ECE(labels_onehot, all_p_star_KLD, bins=30))

#     # print('binary_MCE_SED:', binary_MCE(labels_onehot, all_p_star_SED, bins=30))
#     # print('binary_MCE_L1:', binary_MCE(labels_onehot, all_p_star_L1, bins=30))
#     # print('binary_MCE_KLD:', binary_MCE(labels_onehot, all_p_star_KLD, bins=30))

#     # WARNING!!! 
#     # Inputs for calibrations functions in the shape [n_instance, n_class], [10000, 10]
#     # The pycalib API starts with shape [n_class, n_instance] and use transpose [].T
#     # print('classwise_ECE_SED_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('classwise_MCE_SED_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('conf_ECE_SED_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('conf_MCE_SED_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))  

#     # print('-'*11)

#     # print('classwise_ECE_L1_c:',  np.round(classwise_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('classwise_MCE_L1_c:',  np.round(classwise_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('conf_ECE_L1_c:     ',  np.round(conf_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('conf_MCE_L1_c:     ',  np.round(conf_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_KLD_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('classwise_MCE_KLD_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('conf_ECE_KLD_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('conf_MCE_KLD_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))    

#     # print('-'*22)

#     # print('classwise_ECE_SED_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('classwise_MCE_SED_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('conf_ECE_SED_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('conf_MCE_SED_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_L1_clean:',  np.round(classwise_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('classwise_MCE_L1_clean:',  np.round(classwise_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('conf_ECE_L1_clean:     ',  np.round(conf_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('conf_MCE_L1_clean:     ',  np.round(conf_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_KLD_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('classwise_MCE_KLD_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('conf_ECE_KLD_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('conf_MCE_KLD_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))

#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)

#     kfold_ece_class.append(ece_class)
#     kfold_mce_class.append(mce_class)
#     kfold_ece_confi.append(ece_confi)
#     kfold_mce_confi.append(mce_confi)

# print(np.array(kfold_ece_class))
# print(np.array(kfold_mce_class))
# print(np.array(kfold_ece_confi))
# print(np.array(kfold_mce_confi))

# ece_class_mean = np.round(np.mean(np.array(kfold_ece_class), axis=0), 2)
# mce_class_mean = np.round(np.mean(np.array(kfold_mce_class), axis=0), 2)
# ece_confi_mean = np.round(np.mean(np.array(kfold_ece_confi), axis=0), 2)
# mce_confi_mean = np.round(np.mean(np.array(kfold_mce_confi), axis=0), 2)

# ece_class_std = np.round(np.std(np.array(kfold_ece_class), axis=0), 2)
# mce_class_std = np.round(np.std(np.array(kfold_mce_class), axis=0), 2)
# ece_confi_std = np.round(np.std(np.array(kfold_ece_confi), axis=0), 2)
# mce_confi_std = np.round(np.std(np.array(kfold_mce_confi), axis=0), 2)

# print(f'& {ece_class_mean[0]}\(\pm\){ece_class_std[0]} & {ece_class_mean[1]}\(\pm\){ece_class_std[1]} & {ece_class_mean[2]}\(\pm\){ece_class_std[2]} & {ece_class_mean[3]}\(\pm\){ece_class_std[3]} & {ece_class_mean[4]}\(\pm\){ece_class_std[4]} & {ece_class_mean[5]}\(\pm\){ece_class_std[5]}')

# print(f'& {mce_class_mean[0]}\(\pm\){mce_class_std[0]} & {mce_class_mean[1]}\(\pm\){mce_class_std[1]} & {mce_class_mean[2]}\(\pm\){mce_class_std[2]} & {mce_class_mean[3]}\(\pm\){mce_class_std[3]} & {mce_class_mean[4]}\(\pm\){mce_class_std[4]} & {mce_class_mean[5]}\(\pm\){ece_class_std[5]}')

# print(f'& {ece_confi_mean[0]}\(\pm\){ece_confi_std[0]} & {ece_confi_mean[1]}\(\pm\){ece_confi_std[1]} & {ece_confi_mean[2]}\(\pm\){ece_confi_std[2]} & {ece_confi_mean[3]}\(\pm\){ece_confi_std[3]} & {ece_confi_mean[4]}\(\pm\){ece_confi_std[4]} & {ece_confi_mean[5]}\(\pm\){ece_confi_std[5]}')

# print(f'& {mce_confi_mean[0]}\(\pm\){mce_confi_std[0]} & {mce_confi_mean[1]}\(\pm\){mce_confi_std[1]} & {mce_confi_mean[2]}\(\pm\){mce_confi_std[2]} & {mce_confi_mean[3]}\(\pm\){mce_confi_std[3]} & {mce_confi_mean[4]}\(\pm\){mce_confi_std[4]} & {mce_confi_mean[5]}\(\pm\){ece_confi_std[5]}')

# sys.exit()

# <<< CALIBRATION ERRORS <<<

# >>> SET-VALUED PREDICTIONS >>>

def cal_precise_prediction(all_precise_results, labels):
    # [SED L1 KLD]
    precise_prediction = []

    for i in range(len(all_precise_results)):
        u_alpha = 0

        for j in range(len(labels)):
            if all_precise_results[i][j] == labels[j]:
                u_alpha += 1
            else:
                u_alpha += m_metric[all_precise_results[i][j], labels[j]]

        precise_prediction.append(u_alpha)

    return precise_prediction

def find_a_range(n_class,num_out = 10):
    # Sort rewards descending by column
    m_sorted = np.sort(m_metric, axis=0)[::-1]
    a_list = []

    # iterate each cardinality k
    # k = 1 -> alpha vanishes
    for k in range(2, n_class+1):
        # For each cardinality k, find min s(k)
        s_k_min = np.min(1/np.sum(m_sorted[0:k,:], axis=0))        
        a_list.append((s_k_min * k**2 - 1)/(k - 1))
    
    a_upper = min(a_list)

    # return np.linspace(1.0, a_upper, num=10, endpoint=True), a_list
    return np.linspace(1.0, a_upper, num=num_out, endpoint=True)


def idc(p_star_sensitive, classes, n_class, alpha):
    # Here uses p_star vector multiply m_metric matrix
    # in case m_metric is anti-symetry
    # DONT DO IT HERE
    # p_star_m_metric = np.matmul(p_star, m_metric)
    # p_star_m_metric = p_star
    # Sort descending
    class_order = np.argsort(-p_star_sensitive)

    # class_deque = deque(class_order)
    # max_deque = p_star_m_metric[class_deque.popleft()]
    # min_deque = max_deque
    # sum_deque = max_deque

    # # discount = g_discount(1, max_deque, min_deque, sum_deque)

    # print(type(class_order))
    # print(class_order)
    # print(type(class_deque))
    # print(class_deque)

    # |Y| = 1 => g() = 1
    # sum_tmp = p_star_m_metric[class_order[0]]
    # Singleton prediction
    # eu_max = p_star_m_metric[class_order[0]]
    sum_tmp = 0
    eu_max = 0
    eu_tmp = 0
    k_top = 0

    # For each cardinality
    for k in range(1,n_class+1):
        # utility-discounted g
        g = (alpha / k) + ((1 - alpha) / (k**2))
        # argmax expected value for each cardinality
        # np.sum(p_star_m_metric[class_order[0:k]])
        sum_tmp += p_star_sensitive[class_order[k-1]]
        eu_tmp = g * sum_tmp

        # less than or less than and equal
        if eu_tmp < eu_max:
            break
        else:
            eu_max = eu_tmp
            k_top = k
        
    return list(classes[class_order[0:k_top]])


def cal_set_value_prediction(all_p_star_SED_sensitive,
                                all_p_star_L1_sensitive,
                                all_p_star_KLD_sensitive,
                                classes, n_class, alpha):
    
    alpha_SED_predictions = []
    alpha_L1_predictions = []
    alpha_KLD_predictions = []

    for i in range(len(all_p_star_SED_sensitive)):
        # probability_set = output[:,i,:] #(100,10)

        # >>> Squared Euclidean distance >>>

        alpha_SED = idc(all_p_star_SED_sensitive[i], classes, n_class, alpha)
        alpha_SED_predictions.append(alpha_SED)

        # <<< Squared Euclidean distance <<<
        # >>> L1_distance >>>

        alpha_L1 = idc(all_p_star_L1_sensitive[i], classes, n_class, alpha)
        alpha_L1_predictions.append(alpha_L1)

        # <<< L1_distance <<<

        # >>> KL_divergence >>>

        alpha_KLD = idc(all_p_star_KLD_sensitive[i], classes, n_class, alpha)
        alpha_KLD_predictions.append(alpha_KLD)

        # <<< KL_divergence <<<
    # [[set_SED], [set_L1], [set_KLD]]
    return [alpha_SED_predictions, alpha_L1_predictions, alpha_KLD_predictions]

# >>> Remember to load the right files >>>

train_type = "clean"
# test_type = 'c'
test_type = 'clean'
print(f"train: {train_type} - test: {test_type}")
n_class = 39

# class_to_idx = {
#     'airplane': 0,
#     'automobile': 1,
#     'bird': 2,
#     'cat': 3,
#     'deer': 4,
#     'dog': 5,
#     'frog': 6,
#     'horse': 7,
#     'ship': 8,
#     'truck': 9
#     }
# classes = np.fromiter(class_to_idx.values(), dtype=int)
classes = np.arange(39)
# print(classes)
# sys.exit()
alpha = find_a_range(n_class=n_class, num_out = 20)
# alpha_i = alpha[1]
# alpha_i = 1.6
print(alpha)
# print(alpha_i)

# print(root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/')

 # save_dir = './data/CIFAR-10-C/gaussian_noise_c_fold/'
save_dir = root_dir + f'/data/{data_name}-C/gaussian_{train_type}_fold/Drop_out/'

save_file = f'{save_dir}/set_output/{test_type}_set_value_prediction.txt'
if not os.path.exists(f'{save_dir}/set_output/'):
    print(f"Path does not exist. Creating: {f'{save_dir}set_output/'}")
    os.makedirs(f'{save_dir}set_output/')
    # sys.exit()

with open(save_file, "w") as f:
    f.write(f'list of alpha values: {alpha}')

# kfold_results_c = []
# kfold_results_clean = []
column = [ 'alpha', 'Precise_prediction_SED','Precise_prediction_L1D', 'Precise_prediction_KLD',
            'u65_SED_u65', 'u65_L1D_u65', 'u65_KLD_u65',
            'u65_SED_beta_size', 'u65_L1D_beta_size', 'u65_KLD_beta_size',
            'u65_SED_proportion', 'u65_L1D_proportion', 'u65_KLD_proportion',

            'rec_SED','rec_L1D','rec_KLD',
            'sin_SED','sin_L1D','sin_KLD',
            'set_SED','set_L1D','set_KLD',

            'corr_prec_u65_SED_corr_prec', 'corr_prec_u65_SED_corr_sett', 'inco_prec_u65_SED_corr_sett', 
            'inco_prec_u65_SED_inco_sett', 'inco_prec_u65_SED_inco_prec',

            'corr_prec_u65_L1D_corr_prec', 'corr_prec_u65_L1D_corr_sett', 'inco_prec_u65_L1D_corr_sett',
            'inco_prec_u65_L1D_inco_sett', 'inco_prec_u65_L1D_inco_prec',

            'corr_prec_u65_KLD_corr_prec', 'corr_prec_u65_KLD_corr_sett', 'inco_prec_u65_KLD_corr_sett',
            'inco_prec_u65_KLD_inco_sett', 'inco_prec_u65_KLD_inco_prec']
csv_out = pd.DataFrame(columns= column)

for alpha_i in tqdm(alpha):

    all_precise_prediction_kfold = []

    # [[u65_u65_SED,u65_u80_SED,u80_u80_SED,u80_u65_SED],
    # [u65_u65_L1D,u65_u80_L1D,,u80_u80_L1D,u80_u65_L1D],
    # [u65_u65_KLD,u65_u80_KLD,u80_u80_KLD,u80_u65_KLD], [fold 1], [fold 2]]
    all_set_results_kfold = []

    # [[u65_SED_beta_size, u80_SED_beta_size,
    # u65_L1_beta_size,  u80_L1_beta_size, 
    # u65_KLD_beta_size, u80_KLD_beta_size], [fold 1], [fold 2]]
    beta_size_kfold = []

    # Structure same as beta_size_kfold
    proportion_kfold = []
    
    # The average recall
    rec_kfold = []
    # The proportion of correct singleton predictions
    sin_kfold = []
    # The proportion of correct set-valued predictions
    set_kfold = []
    # [[[corr_prec_u65_SED_corr_prec,
    # corr_prec_u65_SED_corr_sett,
    # inco_prec_u65_SED_corr_sett,
    # inco_prec_u65_SED_inco_sett,
    # inco_prec_u65_SED_inco_prec],
    # [corr_prec_u65_L1D_corr_prec,
    # corr_prec_u65_L1D_corr_sett,
    # inco_prec_u65_L1D_corr_sett,
    # inco_prec_u65_L1D_inco_sett,
    # inco_prec_u65_L1D_inco_prec],
    # [corr_prec_u65_KLD_corr_prec,
    # corr_prec_u65_KLD_corr_sett,
    # inco_prec_u65_KLD_corr_sett,
    # inco_prec_u65_KLD_inco_sett,
    # inco_prec_u65_KLD_inco_prec]], [fold 1], [fold 2]]

    #####################################################

    u65_prec_sett_kfold = []

    # Structure same as u65_prec_sett_kfold
    u80_prec_sett_kfold = []


    for fold in range(k_folds):
        # print(f'FOLD {fold}')
        # print('--------------------------------')

        # fold = 0
        # test_type = 'c'
        # test_type = 'clean'

       
        # with open('./iukm/cifar_test_data_class_to_idx.txt', 'r') as f:
        #     class_to_idx = json.load(f)
        # n_class = len(class_to_idx)
        ###################
        labels = np.load(f'{save_dir}test_labels_{test_type}_fold_{fold}.npy')

        all_p_star_SED = np.load(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy')
        all_p_star_L1 =  np.load(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy')
        all_p_star_KLD = np.load(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy')

        all_precise_results = np.ndarray(shape=(3,len(labels)))

        all_p_star_SED_sensitive = np.matmul(all_p_star_SED, m_metric)
        all_precise_results[0] = np.argmax(all_p_star_SED_sensitive, axis=1)
        
        all_p_star_L1_sensitive = np.matmul(all_p_star_L1, m_metric)
        all_precise_results[1] = np.argmax(all_p_star_L1_sensitive, axis=1)

        all_p_star_KLD_sensitive = np.matmul(all_p_star_KLD, m_metric)
        all_precise_results[2] = np.argmax(all_p_star_KLD_sensitive, axis=1) 
        
        all_precise_prediction_kfold.append(cal_precise_prediction(all_precise_results.astype(int), labels))
        ####################
        # all_precise_prediction_kfold.append(all_precise_results)
        # print((all_precise_results.astype(int)[0] == labels).mean() * 100)

        # all_precise_results = np.ndarray(shape=(3,len(labels)))
        # all_precise_results[0] = np.load(f'{save_dir}predictions_SED_sensitive_{test_type}_fold_{fold}.npy')
        # all_precise_results[1] = np.load(f'{save_dir}predictions_L1_sensitive_{test_type}_fold_{fold}.npy') 
        # all_precise_results[2] = np.load(f'{save_dir}predictions_KLD_sensitive_{test_type}_fold_{fold}.npy')

        # <<< Remember to load the right files <<<

        # class_name = list(class_to_idx.keys())

        # [[alpha_SED], [alpha_L1], [alpha_KLD]]        
        all_set_predictions = cal_set_value_prediction(all_p_star_SED_sensitive,
                                                        all_p_star_L1_sensitive,
                                                        all_p_star_KLD_sensitive,
                                                        classes, n_class, alpha_i)
        
        # save_file_all_set_predictions = f'{save_dir}all_set_predictions.pickle'
        # # a = {'hello': 'world'}
        # with open(save_file_all_set_predictions, 'wb') as f:
        #     pickle.dump(all_set_predictions, f)

        # with open(save_file_all_set_predictions, 'rb') as floaded:
        #     b = pickle.load(floaded)

        # sys.exit()

        # [[u65_u65, u65_u80, u80_u80, u80_u65], [L1], [KLD]]
        # all_set_results = np.ndarray(shape=(3,4))
        all_set_results = np.ndarray(shape=(3,4))

        ## [[SED], [L1], [KLD]]
        ## [[u65_u65_uni, u65_u65_mul, 
        ##   u80_u80_uni, u80_u80_mul], [L1], [KLD]]
        all_set_count = np.zeros([3, 4])
        # [[precise_u65_uni, precise_u65_mul,
        #   precise_u80_uni, precise_u80_mul], [L1], [KLD]]
        all_precise_count = np.zeros([3, 4])
        # [[SED], [L1], [KLD]]
        # [[u65_set_count, u65_element_count,
        #   u80_set_count, u80_element_count], [L1], [KLD]]
        avg_beta_size = np.zeros([3, 4])


        # >>> From set-valued perspective >>>
        # j=1=set_SED, j=2=set_L1, j=3=set_KLD
        for j in range(len(all_set_predictions)):
            u_alpha_total = 0
            for i in range(len(labels)):
                u_alpha = 0
                u_alpha_prediction = all_set_predictions[j][i]

                # singleton prediction
                if len(u_alpha_prediction) == 1:
                    #if u_alpha_prediction[0] == labels[i]:
                    u_alpha += m_metric[u_alpha_prediction[0], labels[i]]
                    all_set_count[j][0] += 1

                    if all_precise_results[j][i] == labels[i]:
                        all_precise_count[j][0] += 1
                # set prediction
                else:
                    all_set_count[j][1] += 1
                    avg_beta_size[j][0] += 1
                    avg_beta_size[j][1] += len(u_alpha_prediction)

                    for predicted_element in u_alpha_prediction:
                        u_alpha += m_metric[predicted_element, labels[i]] 

                    u_alpha = u_alpha * (alpha_i/len(u_alpha_prediction) + (1 - alpha_i)/(len(u_alpha_prediction)**2))
                    # if labels[i] in u_alpha_prediction:
                    #     # u_alpha += 1
                    #     u_alpha += alpha_i/len(u_alpha_prediction) + (1 - alpha_i)/(len(u_alpha_prediction)**2)
                    #     # u65_u65 += (-0.6/(len(u65_prediction)**2) + 1.6/len(u65_prediction))
                    #     # u65_u80 += (-1.2/(len(u65_prediction)**2) + 2.2/len(u65_prediction))
                        
                    
                    if all_precise_results[j][i] == labels[i]:
                        all_precise_count[j][1] += 1
                
                u_alpha_total += u_alpha
                if u_alpha < 0 or u_alpha > 1:
                    print(u_alpha)
                    sys.exit()
            
            all_set_results[j][0] = u_alpha_total


        all_set_results_kfold.append((all_set_results/len(labels))*100)
        # all_set_results = np.round((all_set_results/len(labels))*100, 2)
        all_set_count /= len(labels)
        all_precise_count /= len(labels) # recal

        u65_SED_beta_size = avg_beta_size[0][1]/avg_beta_size[0][0]
        # u80_SED_beta_size = avg_beta_size[0][3]/avg_beta_size[0][2]
        u65_L1_beta_size =  avg_beta_size[1][1]/avg_beta_size[1][0]
        # u80_L1_beta_size =  avg_beta_size[1][3]/avg_beta_size[1][2]
        u65_KLD_beta_size = avg_beta_size[2][1]/avg_beta_size[2][0]
        # u80_KLD_beta_size = avg_beta_size[2][3]/avg_beta_size[2][2]

        # beta_size_kfold.append([u65_SED_beta_size,
        #                         u80_SED_beta_size,
        #                         u65_L1_beta_size ,
        #                         u80_L1_beta_size ,
        #                         u65_KLD_beta_size,
        #                         u80_KLD_beta_size])

        beta_size_kfold.append([u65_SED_beta_size,
                                # u80_SED_beta_size,
                                u65_L1_beta_size ,
                                # u80_L1_beta_size ,
                                u65_KLD_beta_size,
                                # u80_KLD_beta_size
                                ])

        u65_SED_proportion = (avg_beta_size[0][0]/len(labels))*100
        # u80_SED_proportion = (avg_beta_size[0][2]/len(labels))*100
        u65_L1_proportion =  (avg_beta_size[1][0]/len(labels))*100
        # u80_L1_proportion =  (avg_beta_size[1][2]/len(labels))*100
        u65_KLD_proportion = (avg_beta_size[2][0]/len(labels))*100
        # u80_KLD_proportion = (avg_beta_size[2][2]/len(labels))*100

        # proportion_kfold.append([u65_SED_proportion,
        #                          u80_SED_proportion,
        #                          u65_L1_proportion ,
        #                          u80_L1_proportion ,
        #                          u65_KLD_proportion,
        #                          u80_KLD_proportion])

        proportion_kfold.append([u65_SED_proportion,
                                #  u80_SED_proportion,
                                u65_L1_proportion ,
                                #  u80_L1_proportion ,
                                u65_KLD_proportion,
                                #  u80_KLD_proportion
                                ])



        # <<< Set size <<<

        # <<< From set-valued perspective <<<

        # >>> from precise perspective >>>        

        # [[SED], [L1], [KLD]]
        # [[precise_correct, precise_incorrect], [L1], [KLD]]
        precise_cor_inc_count = np.zeros([3, 2])
        # precise       precise_correct*         precise_incorrect*
        # set_valued    0 precise_correct*       2 precise_incorrect
        #               1 set_correct            3 set_correct*
        #                                        4 set_incorrect
        # [[SED], [L1], [KLD]]
        # [[u65_precise_correct, u65_set_correct, 
        #   u65_precise_incorrect, u65_set_correct, u65_set_incorrect,
        #   u80_precise_correct, u80_set_correct, 
        #   u80_precise_incorrect, u80_set_correct, u80_set_incorrect],
        #   [L1], [KLD]]

        # set_cor_inc_count = np.zeros([3, 10])
        set_cor_inc_count = np.zeros([3, 5])

        # all_set_predictions = [[u65_SED_predictions, u80_SED_predictions],
        #                        [u65_L1_predictions, u80_L1_predictions],
        #                        [u65_KLD_predictions, u80_KLD_predictions]]
        # If True label = 2:
        # Precise prediction = 1
        # u65 = [1], singleton --> pre_inc_u65_sing_incor + 1
        # u80 = [1, 3], set    --> pre_inc_u80_set_incor + 1
        # Therefore: pre_inc_u80_set_incor larger than pre_inc_u65_set_incor
        for j in range(len(all_precise_results)):
            for i in range(len(labels)):
                # precise + correct
                if all_precise_results[j][i] == labels[i]:
                    precise_cor_inc_count[j][0] += 1
                    # u65
                    # [[alpha_SED], [alpha_L1], [alpha_KLD]]
                    if len(all_set_predictions[j][i]) == 1:
                        set_cor_inc_count[j][0] += 1
                    else:
                        set_cor_inc_count[j][1] += 1
                                
                else: # precise + incorrect
                    precise_cor_inc_count[j][1] += 1
                    # u65
                    if len(all_set_predictions[j][i]) == 1:                
                        set_cor_inc_count[j][2] += 1
                    elif labels[i] not in all_set_predictions[j][i]:
                        set_cor_inc_count[j][4] += 1
                    else:
                        set_cor_inc_count[j][3] += 1
                    

        precise_cor_inc_count_labels = precise_cor_inc_count / len(labels)
        set_cor_inc_count_labels = set_cor_inc_count / len(labels)

        # [[SED], [L1], [KLD]]
        # [[u65_precise_correct, u65_set_correct, 
        #   u65_precise_incorrect, u65_set_correct, u65_set_incorrect,
        #   u80_precise_correct, u80_set_correct, 
        #   u80_precise_incorrect, u80_set_correct, u80_set_incorrect],
        #   [L1], [KLD]]
        # [[precise_correct, precise_incorrect], [L1], [KLD]]
        corr_prec_u65_SED_corr_prec = (set_cor_inc_count[0][0]/precise_cor_inc_count[0][0])*100
        corr_prec_u65_SED_corr_sett = (set_cor_inc_count[0][1]/precise_cor_inc_count[0][0])*100
        inco_prec_u65_SED_corr_sett = (set_cor_inc_count[0][3]/precise_cor_inc_count[0][1])*100
        inco_prec_u65_SED_inco_sett = (set_cor_inc_count[0][4]/precise_cor_inc_count[0][1])*100
        inco_prec_u65_SED_inco_prec = (set_cor_inc_count[0][2]/precise_cor_inc_count[0][1])*100
        
        corr_prec_u65_L1_corr_prec =  (set_cor_inc_count[1][0]/precise_cor_inc_count[1][0])*100
        corr_prec_u65_L1_corr_sett =  (set_cor_inc_count[1][1]/precise_cor_inc_count[1][0])*100
        inco_prec_u65_L1_corr_sett =  (set_cor_inc_count[1][3]/precise_cor_inc_count[1][1])*100
        inco_prec_u65_L1_inco_sett =  (set_cor_inc_count[1][4]/precise_cor_inc_count[1][1])*100
        inco_prec_u65_L1_inco_prec =  (set_cor_inc_count[1][2]/precise_cor_inc_count[1][1])*100
        
        corr_prec_u65_KLD_corr_prec = (set_cor_inc_count[2][0]/precise_cor_inc_count[2][0])*100
        corr_prec_u65_KLD_corr_sett = (set_cor_inc_count[2][1]/precise_cor_inc_count[2][0])*100
        inco_prec_u65_KLD_corr_sett = (set_cor_inc_count[2][3]/precise_cor_inc_count[2][1])*100
        inco_prec_u65_KLD_inco_sett = (set_cor_inc_count[2][4]/precise_cor_inc_count[2][1])*100
        inco_prec_u65_KLD_inco_prec = (set_cor_inc_count[2][2]/precise_cor_inc_count[2][1])*100

        u65_prec_sett_kfold.append([[corr_prec_u65_SED_corr_prec,
                                    corr_prec_u65_SED_corr_sett,
                                    inco_prec_u65_SED_corr_sett,
                                    inco_prec_u65_SED_inco_sett,
                                    inco_prec_u65_SED_inco_prec],
                                    [corr_prec_u65_L1_corr_prec,
                                    corr_prec_u65_L1_corr_sett,
                                    inco_prec_u65_L1_corr_sett,
                                    inco_prec_u65_L1_inco_sett,
                                    inco_prec_u65_L1_inco_prec],
                                    [corr_prec_u65_KLD_corr_prec,
                                    corr_prec_u65_KLD_corr_sett,
                                    inco_prec_u65_KLD_corr_sett,
                                    inco_prec_u65_KLD_inco_sett,
                                    inco_prec_u65_KLD_inco_prec]])


        #new 
        # the average recall
        
        rec_SED = precise_cor_inc_count_labels[0][0] + set_cor_inc_count_labels[0][3]
        rec_L1D = precise_cor_inc_count_labels[1][0] + set_cor_inc_count_labels[1][3]
        rec_KLD = precise_cor_inc_count_labels[2][0] + set_cor_inc_count_labels[2][3]

        rec_kfold.append([rec_SED*100,rec_L1D*100,rec_KLD*100])
        # the proportion of corect singletion:

        sin_SED = set_cor_inc_count[0][0] / (set_cor_inc_count[0][0] + set_cor_inc_count[0][2]) 
        sin_L1D = set_cor_inc_count[1][0] / (set_cor_inc_count[1][0] + set_cor_inc_count[1][2]) 
        sin_KLD = set_cor_inc_count[2][0] / (set_cor_inc_count[2][0] + set_cor_inc_count[2][2]) 

        sin_kfold.append([sin_SED*100,sin_L1D*100,sin_KLD*100])
        # the proportion of corect singletion:

        set_SED = (set_cor_inc_count[0][1] + set_cor_inc_count[0][3]) / (set_cor_inc_count[0][1] + set_cor_inc_count[0][3] + set_cor_inc_count[0][4]) 
        set_L1D = (set_cor_inc_count[1][1] + set_cor_inc_count[1][3]) / (set_cor_inc_count[1][1] + set_cor_inc_count[1][3] + set_cor_inc_count[1][4])
        set_KLD = (set_cor_inc_count[2][1] + set_cor_inc_count[2][3]) / (set_cor_inc_count[2][1] + set_cor_inc_count[2][3] + set_cor_inc_count[2][4])

        set_kfold.append([set_SED*100,set_L1D*100,set_KLD*100])
        # corr_prec_u80_SED_corr_prec = (set_cor_inc_count[0][5]/precise_cor_inc_count[0][0])*100
        # corr_prec_u80_SED_corr_sett = (set_cor_inc_count[0][6]/precise_cor_inc_count[0][0])*100
        # inco_prec_u80_SED_corr_sett = (set_cor_inc_count[0][8]/precise_cor_inc_count[0][1])*100
        # inco_prec_u80_SED_inco_sett = (set_cor_inc_count[0][9]/precise_cor_inc_count[0][1])*100
        # inco_prec_u80_SED_inco_prec = (set_cor_inc_count[0][7]/precise_cor_inc_count[0][1])*100
        
        # corr_prec_u80_L1_corr_prec =  (set_cor_inc_count[1][5]/precise_cor_inc_count[1][0])*100
        # corr_prec_u80_L1_corr_sett =  (set_cor_inc_count[1][6]/precise_cor_inc_count[1][0])*100
        # inco_prec_u80_L1_corr_sett =  (set_cor_inc_count[1][8]/precise_cor_inc_count[1][1])*100
        # inco_prec_u80_L1_inco_sett =  (set_cor_inc_count[1][9]/precise_cor_inc_count[1][1])*100
        # inco_prec_u80_L1_inco_prec =  (set_cor_inc_count[1][7]/precise_cor_inc_count[1][1])*100
        
        # corr_prec_u80_KLD_corr_prec = (set_cor_inc_count[2][5]/precise_cor_inc_count[2][0])*100
        # corr_prec_u80_KLD_corr_sett = (set_cor_inc_count[2][6]/precise_cor_inc_count[2][0])*100
        # inco_prec_u80_KLD_corr_sett = (set_cor_inc_count[2][8]/precise_cor_inc_count[2][1])*100
        # inco_prec_u80_KLD_inco_sett = (set_cor_inc_count[2][9]/precise_cor_inc_count[2][1])*100
        # inco_prec_u80_KLD_inco_prec = (set_cor_inc_count[2][7]/precise_cor_inc_count[2][1])*100

        # u80_prec_sett_kfold.append([[corr_prec_u80_SED_corr_prec,
        #                             corr_prec_u80_SED_corr_sett,
        #                             inco_prec_u80_SED_corr_sett,
        #                             inco_prec_u80_SED_inco_sett,
        #                             inco_prec_u80_SED_inco_prec],
        #                             [corr_prec_u80_L1_corr_prec,
        #                             corr_prec_u80_L1_corr_sett,
        #                             inco_prec_u80_L1_corr_sett,
        #                             inco_prec_u80_L1_inco_sett,
        #                             inco_prec_u80_L1_inco_prec],
        #                             [corr_prec_u80_KLD_corr_prec,
        #                             corr_prec_u80_KLD_corr_sett,
        #                             inco_prec_u80_KLD_corr_sett,
        #                             inco_prec_u80_KLD_inco_sett,
        #                             inco_prec_u80_KLD_inco_prec]])


    # >>> Precise prediction mean and std >>>

    all_precise_prediction_kfold = np.array(all_precise_prediction_kfold)/len(labels)*100
    precise_prediction_mean = np.round(np.mean(all_precise_prediction_kfold, axis=0),2)
    precise_prediction_std = np.round(np.std(all_precise_prediction_kfold, axis=0),2)

    # <<< Precise prediction mean and std <<<

    # >>> Set predictions mean and std >>>

    all_set_results_mean = np.round(np.mean(all_set_results_kfold, axis=0), 2)
    all_set_results_std  = np.round(np.std(all_set_results_kfold, axis=0), 2)

    # <<< Set prediction mean and std <<<

    # >>> Beta size and proportion mean and std >>>

    beta_size_mean = np.round(np.mean(beta_size_kfold, axis=0), 2)
    beta_size_std  = np.round(np.std(beta_size_kfold, axis=0), 2)
    proportion_mean = np.round(np.mean(proportion_kfold, axis=0), 2)
    proportion_std = np.round(np.std(proportion_kfold, axis=0), 2)

    # <<< Beta size and proportion mean and std <<<

    # >>> Precise and set correct and incorrect >>>

    u65_prec_sett_mean = np.round(np.mean(u65_prec_sett_kfold, axis=0), 2)
    u65_prec_sett_std  = np.round(np.std(u65_prec_sett_kfold, axis=0), 2)

    # u80_prec_sett_mean = np.round(np.mean(u80_prec_sett_kfold, axis=0), 2)
    # u80_prec_sett_std  = np.round(np.std(u80_prec_sett_kfold, axis=0), 2)

    # <<< Precise and set correct and incorrect <<<

    rec_mean = np.round(np.mean(rec_kfold, axis=0), 2)
    rec_std = np.round(np.std(rec_kfold, axis=0), 2)

    ####

    sin_mean = np.round(np.mean(sin_kfold, axis=0), 2)
    sin_std = np.round(np.std(sin_kfold, axis=0), 2)

    ####

    set_mean = np.round(np.mean(set_kfold, axis=0), 2)
    set_std = np.round(np.std(set_kfold, axis=0), 2)

    #### save csv file
    temp_pandas = []
    temp_pandas.append(np.round(alpha_i, 5))
    temp_pandas.append(f"{precise_prediction_mean[0]} : {precise_prediction_std[0]}")
    temp_pandas.append(f"{precise_prediction_mean[1]} : {precise_prediction_std[1]}")
    temp_pandas.append(f"{precise_prediction_mean[2]} : {precise_prediction_std[2]}")
    # u65_SED_u65
    temp_pandas.append(f"{all_set_results_mean[0][0]} : {all_set_results_std[0][0]}")
    temp_pandas.append(f"{all_set_results_mean[1][0]} : {all_set_results_std[1][0]}")
    temp_pandas.append(f"{all_set_results_mean[2][0]} : {all_set_results_std[2][0]}")
    # u65_SED_beta_size
    temp_pandas.append(f"{beta_size_mean[0]} : {beta_size_std[0]}")
    temp_pandas.append(f"{beta_size_mean[1]} : {beta_size_std[1]}")
    temp_pandas.append(f"{beta_size_mean[2]} : {beta_size_std[2]}")
    # u65_SED_proportion
    temp_pandas.append(f"{proportion_mean[0]} : {proportion_std[0]}")
    temp_pandas.append(f"{proportion_mean[1]} : {proportion_std[1]}")
    temp_pandas.append(f"{proportion_mean[2]} : {proportion_std[2]}")
    # rec
    temp_pandas.append(f"{rec_mean[0]} : {rec_std[0]}")
    temp_pandas.append(f"{rec_mean[1]} : {rec_std[1]}")
    temp_pandas.append(f"{rec_mean[2]} : {rec_std[2]}")
    # sin
    temp_pandas.append(f"{sin_mean[0]} : {sin_std[0]}")
    temp_pandas.append(f"{sin_mean[1]} : {sin_std[1]}")
    temp_pandas.append(f"{sin_mean[2]} : {sin_std[2]}")
    # set
    temp_pandas.append(f"{set_mean[0]} : {set_std[0]}")
    temp_pandas.append(f"{set_mean[1]} : {set_std[1]}")
    temp_pandas.append(f"{set_mean[2]} : {set_std[2]}")
    # Precise and set correct and incorrec
    # # SED
    temp_pandas.append(f"{u65_prec_sett_mean[0][0]} : {u65_prec_sett_std[0][0]}")
    temp_pandas.append(f"{u65_prec_sett_mean[0][1]} : {u65_prec_sett_std[0][1]}")
    temp_pandas.append(f"{u65_prec_sett_mean[0][2]} : {u65_prec_sett_std[0][2]}")
    temp_pandas.append(f"{u65_prec_sett_mean[0][3]} : {u65_prec_sett_std[0][3]}")
    temp_pandas.append(f"{u65_prec_sett_mean[0][4]} : {u65_prec_sett_std[0][4]}")
    # # L1D
    temp_pandas.append(f"{u65_prec_sett_mean[1][0]} : {u65_prec_sett_std[1][0]}")
    temp_pandas.append(f"{u65_prec_sett_mean[1][1]} : {u65_prec_sett_std[1][1]}")
    temp_pandas.append(f"{u65_prec_sett_mean[1][2]} : {u65_prec_sett_std[1][2]}")
    temp_pandas.append(f"{u65_prec_sett_mean[1][3]} : {u65_prec_sett_std[1][3]}")
    temp_pandas.append(f"{u65_prec_sett_mean[1][4]} : {u65_prec_sett_std[1][4]}")
    # # KLD
    temp_pandas.append(f"{u65_prec_sett_mean[2][0]} : {u65_prec_sett_std[2][0]}")
    temp_pandas.append(f"{u65_prec_sett_mean[2][1]} : {u65_prec_sett_std[2][1]}")
    temp_pandas.append(f"{u65_prec_sett_mean[2][2]} : {u65_prec_sett_std[2][2]}")
    temp_pandas.append(f"{u65_prec_sett_mean[2][3]} : {u65_prec_sett_std[2][3]}")
    temp_pandas.append(f"{u65_prec_sett_mean[2][4]} : {u65_prec_sett_std[2][4]}")
    ####

    csv_out.loc[len(csv_out)] = temp_pandas

    #### save txt file
    with open(save_file, "a") as f:
        f.write(3*'\n')
        
        f.write(10*'*'+ f" {alpha_i} " + 10*"*")

        f.write(3*'\n')
        f.write('\n>>> Precise predictions >>>')
        f.write(f'\nPrecise_prediction_SED_mean: {precise_prediction_mean[0]}')
        f.write(f'\nPrecise_prediction_L1D_mean: {precise_prediction_mean[1]}')
        f.write(f'\nPrecise_prediction_KLD_mean: {precise_prediction_mean[2]}')
        f.write(f'\nPrecise_prediction_SED_std: {precise_prediction_std[0]}')
        f.write(f'\nPrecise_prediction_L1D_std: {precise_prediction_std[1]}')
        f.write(f'\nPrecise_prediction_KLD_std: {precise_prediction_std[2]}')
        f.write('\n<<< Precise predictions <<<')
        f.write('\n>>> Set predictions >>>')
        f.write(f'\nu65_SED_u65_mean: {all_set_results_mean[0][0]}')
        # f.write(f'\nu65_SED_u80_mean: {all_set_results_mean[0][1]}')
        # f.write(f'\nu80_SED_u80_mean: {all_set_results_mean[0][2]}')
        # f.write(f'\nu80_SED_u65_mean: {all_set_results_mean[0][3]}')
        f.write(f'\nu65_L1D_u65_mean: {all_set_results_mean[1][0]}')
        # f.write(f'\nu65_L1D_u80_mean: {all_set_results_mean[1][1]}')
        # f.write(f'\nu80_L1D_u80_mean: {all_set_results_mean[1][2]}')
        # f.write(f'\nu80_L1D_u65_mean: {all_set_results_mean[1][3]}')
        f.write(f'\nu65_KLD_u65_mean: {all_set_results_mean[2][0]}')
        # f.write(f'\nu65_KLD_u80_mean: {all_set_results_mean[2][1]}')
        # f.write(f'\nu80_KLD_u80_mean: {all_set_results_mean[2][2]}')
        # f.write(f'\nu80_KLD_u65_mean: {all_set_results_mean[2][3]}')
        f.write(f'\nu65_SED_u65_std: {all_set_results_std[0][0]}')
        # f.write(f'\nu65_SED_u80_std: {all_set_results_std[0][1]}')
        # f.write(f'\nu80_SED_u80_std: {all_set_results_std[0][2]}')
        # f.write(f'\nu80_SED_u65_std: {all_set_results_std[0][3]}')
        f.write(f'\nu65_L1D_u65_std: {all_set_results_std[1][0]}')
        # f.write(f'\nu65_L1D_u80_std: {all_set_results_std[1][1]}')
        # f.write(f'\nu80_L1D_u80_std: {all_set_results_std[1][2]}')
        # f.write(f'\nu80_L1D_u65_std: {all_set_results_std[1][3]}')
        f.write(f'\nu65_KLD_u65_std: {all_set_results_std[2][0]}')
        # f.write(f'\nu65_KLD_u80_std: {all_set_results_std[2][1]}')
        # f.write(f'\nu80_KLD_u80_std: {all_set_results_std[2][2]}')
        # f.write(f'\nu80_KLD_u65_std: {all_set_results_std[2][3]}')
        f.write('\n<<< Set predictions <<<')
        f.write('\n>>> Beta size >>>')
        f.write(f'\nu65_SED_beta_size_mean: {beta_size_mean[0]}')
        # f.write(f'\nu80_SED_beta_size_mean: {beta_size_mean[1]}')
        f.write(f'\nu65_L1D_beta_size_mean: {beta_size_mean[1]}')
        # f.write(f'\nu80_L1D_beta_size_mean: {beta_size_mean[3]}')
        f.write(f'\nu65_KLD_beta_size_mean: {beta_size_mean[2]}')
        # f.write(f'\nu80_KLD_beta_size_mean: {beta_size_mean[5]}')
        f.write(f'\nu65_SED_beta_size_std: {beta_size_std[0]}')
        # f.write(f'\nu80_SED_beta_size_std: {beta_size_std[1]}')
        f.write(f'\nu65_L1D_beta_size_std: {beta_size_std[1]}')
        # f.write(f'\nu80_L1D_beta_size_std: {beta_size_std[3]}')
        f.write(f'\nu65_KLD_beta_size_std: {beta_size_std[2]}')
        # f.write(f'\nu80_KLD_beta_size_std: {beta_size_std[5]}')
        f.write('\n<<< Beta size <<<')
        f.write('\n>>> Proportion >>>')
        f.write(f'\nu65_SED_proportion_mean: {proportion_mean[0]}')
        # f.write(f'\nu80_SED_proportion_mean: {proportion_mean[1]}')
        f.write(f'\nu65_L1D_proportion_mean: {proportion_mean[1]}')
        # f.write(f'\nu80_L1D_proportion_mean: {proportion_mean[3]}')
        f.write(f'\nu65_KLD_proportion_mean: {proportion_mean[2]}')
        # f.write(f'\nu80_KLD_proportion_mean: {proportion_mean[5]}')
        f.write(f'\nu65_SED_proportion_std: {proportion_std[0]}')
        # f.write(f'\nu80_SED_proportion_std: {proportion_std[1]}')
        f.write(f'\nu65_L1D_proportion_std: {proportion_std[1]}')
        # f.write(f'\nu80_L1D_proportion_std: {proportion_std[3]}')
        f.write(f'\nu65_KLD_proportion_std: {proportion_std[2]}')
        # f.write(f'\nu80_KLD_proportion_std: {proportion_std[5]}')
        f.write('\n<<< Proportion <<<')
        f.write('\n>>> rec >>>')
        f.write(f'\nu65_SED_rec_mean: {rec_mean[0]}')
        f.write(f'\nu65_L1D_rec_mean: {rec_mean[1]}')
        f.write(f'\nu65_KLD_rec_mean: {rec_mean[2]}')
        f.write(f'\nu65_SED_rec_std: {rec_std[0]}')
        f.write(f'\nu65_L1D_rec_std: {rec_std[1]}')
        f.write(f'\nu65_KLD_rec_std: {rec_std[2]}')
        f.write('\n<<< rec <<<')
        f.write('\n>>> sin >>>')
        f.write(f'\nu65_SED_sin_mean: {sin_mean[0]}')
        f.write(f'\nu65_L1D_sin_mean: {sin_mean[1]}')
        f.write(f'\nu65_KLD_sin_mean: {sin_mean[2]}')
        f.write(f'\nu65_SED_sin_std: {sin_std[0]}')
        f.write(f'\nu65_L1D_sin_std: {sin_std[1]}')
        f.write(f'\nu65_KLD_sin_std: {sin_std[2]}')
        f.write('\n<<< sin <<<')
        f.write('\n>>> set >>>')
        f.write(f'\nu65_SED_set_mean: {set_mean[0]}')
        f.write(f'\nu65_L1D_set_mean: {set_mean[1]}')
        f.write(f'\nu65_KLD_set_mean: {set_mean[2]}')
        f.write(f'\nu65_SED_set_std: {set_std[0]}')
        f.write(f'\nu65_L1D_set_std: {set_std[1]}')
        f.write(f'\nu65_KLD_set_std: {set_std[2]}')
        f.write('\n<<< set <<<')
        f.write('\n>>> Precise and set correct and incorrect >>>')
        f.write(f'\ncorr_prec_u65_SED_corr_prec_mean: {u65_prec_sett_mean[0][0]}')
        f.write(f'\ncorr_prec_u65_SED_corr_sett_mean: {u65_prec_sett_mean[0][1]}')
        f.write(f'\ninco_prec_u65_SED_corr_sett_mean: {u65_prec_sett_mean[0][2]}')
        f.write(f'\ninco_prec_u65_SED_inco_sett_mean: {u65_prec_sett_mean[0][3]}')
        f.write(f'\ninco_prec_u65_SED_inco_prec_mean: {u65_prec_sett_mean[0][4]}')
        f.write(f'\ncorr_prec_u65_L1_corr_prec_mean:  {u65_prec_sett_mean[1][0]}')
        f.write(f'\ncorr_prec_u65_L1_corr_sett_mean:  {u65_prec_sett_mean[1][1]}')
        f.write(f'\ninco_prec_u65_L1_corr_sett_mean:  {u65_prec_sett_mean[1][2]}')
        f.write(f'\ninco_prec_u65_L1_inco_sett_mean:  {u65_prec_sett_mean[1][3]}')
        f.write(f'\ninco_prec_u65_L1_inco_prec_mean:  {u65_prec_sett_mean[1][4]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_prec_mean: {u65_prec_sett_mean[2][0]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_sett_mean: {u65_prec_sett_mean[2][1]}')
        f.write(f'\ninco_prec_u65_KLD_corr_sett_mean: {u65_prec_sett_mean[2][2]}')
        f.write(f'\ninco_prec_u65_KLD_inco_sett_mean: {u65_prec_sett_mean[2][3]}')
        f.write(f'\ninco_prec_u65_KLD_inco_prec_mean: {u65_prec_sett_mean[2][4]}')
        # f.write(f'\ncorr_prec_u80_SED_corr_prec_mean: {u80_prec_sett_mean[0][0]}')
        # f.write(f'\ncorr_prec_u80_SED_corr_sett_mean: {u80_prec_sett_mean[0][1]}')
        # f.write(f'\ninco_prec_u80_SED_corr_sett_mean: {u80_prec_sett_mean[0][2]}')
        # f.write(f'\ninco_prec_u80_SED_inco_sett_mean: {u80_prec_sett_mean[0][3]}')
        # f.write(f'\ninco_prec_u80_SED_inco_prec_mean: {u80_prec_sett_mean[0][4]}')
        # f.write(f'\ncorr_prec_u80_L1_corr_prec_mean:  {u80_prec_sett_mean[1][0]}')
        # f.write(f'\ncorr_prec_u80_L1_corr_sett_mean:  {u80_prec_sett_mean[1][1]}')
        # f.write(f'\ninco_prec_u80_L1_corr_sett_mean:  {u80_prec_sett_mean[1][2]}')
        # f.write(f'\ninco_prec_u80_L1_inco_sett_mean:  {u80_prec_sett_mean[1][3]}')
        # f.write(f'\ninco_prec_u80_L1_inco_prec_mean:  {u80_prec_sett_mean[1][4]}')
        # f.write(f'\ncorr_prec_u80_KLD_corr_prec_mean: {u80_prec_sett_mean[2][0]}')
        # f.write(f'\ncorr_prec_u80_KLD_corr_sett_mean: {u80_prec_sett_mean[2][1]}')
        # f.write(f'\ninco_prec_u80_KLD_corr_sett_mean: {u80_prec_sett_mean[2][2]}')
        # f.write(f'\ninco_prec_u80_KLD_inco_sett_mean: {u80_prec_sett_mean[2][3]}')
        # f.write(f'\ninco_prec_u80_KLD_inco_prec_mean: {u80_prec_sett_mean[2][4]}')
        f.write(f'\ncorr_prec_u65_SED_corr_prec_std:  {u65_prec_sett_std[0][0]}')
        f.write(f'\ncorr_prec_u65_SED_corr_sett_std:  {u65_prec_sett_std[0][1]}')
        f.write(f'\ninco_prec_u65_SED_corr_sett_std:  {u65_prec_sett_std[0][2]}')
        f.write(f'\ninco_prec_u65_SED_inco_sett_std:  {u65_prec_sett_std[0][3]}')
        f.write(f'\ninco_prec_u65_SED_inco_prec_std:  {u65_prec_sett_std[0][4]}') 
        f.write(f'\ncorr_prec_u65_L1_corr_prec_std:   {u65_prec_sett_std[1][0]}')
        f.write(f'\ncorr_prec_u65_L1_corr_sett_std:   {u65_prec_sett_std[1][1]}')
        f.write(f'\ninco_prec_u65_L1_corr_sett_std:   {u65_prec_sett_std[1][2]}')
        f.write(f'\ninco_prec_u65_L1_inco_sett_std:   {u65_prec_sett_std[1][3]}')
        f.write(f'\ninco_prec_u65_L1_inco_prec_std:   {u65_prec_sett_std[1][4]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_prec_std:  {u65_prec_sett_std[2][0]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_sett_std:  {u65_prec_sett_std[2][1]}')
        f.write(f'\ninco_prec_u65_KLD_corr_sett_std:  {u65_prec_sett_std[2][2]}')
        f.write(f'\ninco_prec_u65_KLD_inco_sett_std:  {u65_prec_sett_std[2][3]}')
        f.write(f'\ninco_prec_u65_KLD_inco_prec_std:  {u65_prec_sett_std[2][4]}')
        # f.write(f'\ncorr_prec_u80_SED_corr_prec_std:  {u80_prec_sett_std[0][0]}')
        # f.write(f'\ncorr_prec_u80_SED_corr_sett_std:  {u80_prec_sett_std[0][1]}')
        # f.write(f'\ninco_prec_u80_SED_corr_sett_std:  {u80_prec_sett_std[0][2]}')
        # f.write(f'\ninco_prec_u80_SED_inco_sett_std:  {u80_prec_sett_std[0][3]}')
        # f.write(f'\ninco_prec_u80_SED_inco_prec_std:  {u80_prec_sett_std[0][4]}')
        # f.write(f'\ncorr_prec_u80_L1_corr_prec_std:   {u80_prec_sett_std[1][0]}')
        # f.write(f'\ncorr_prec_u80_L1_corr_sett_std:   {u80_prec_sett_std[1][1]}')
        # f.write(f'\ninco_prec_u80_L1_corr_sett_std:   {u80_prec_sett_std[1][2]}')
        # f.write(f'\ninco_prec_u80_L1_inco_sett_std:   {u80_prec_sett_std[1][3]}')
        # f.write(f'\ninco_prec_u80_L1_inco_prec_std:   {u80_prec_sett_std[1][4]}')
        # f.write(f'\ncorr_prec_u80_KLD_corr_prec_std:  {u80_prec_sett_std[2][0]}')
        # f.write(f'\ncorr_prec_u80_KLD_corr_sett_std:  {u80_prec_sett_std[2][1]}')
        # f.write(f'\ninco_prec_u80_KLD_corr_sett_std:  {u80_prec_sett_std[2][2]}')
        # f.write(f'\ninco_prec_u80_KLD_inco_sett_std:  {u80_prec_sett_std[2][3]}')
        # f.write(f'\ninco_prec_u80_KLD_inco_prec_std:  {u80_prec_sett_std[2][4]}')
        f.write('\n<<< Precise and set correct and incorrect <<<')

csv_out = csv_out.T
csv_out.to_csv(f'{save_dir}/set_output/{test_type}_set_value_prediction.csv', header=False)

# print('corr_prec_u65_SED_corr_prec:', corr_prec_u65_SED_corr_prec)
# print('corr_prec_u65_SED_corr_sett:', corr_prec_u65_SED_corr_sett)
# print('inco_prec_u65_SED_corr_sett:', inco_prec_u65_SED_corr_sett)
# print('inco_prec_u65_SED_inco_sett:', inco_prec_u65_SED_inco_sett)
# print('inco_prec_u65_SED_inco_prec:', inco_prec_u65_SED_inco_prec)

# print('corr_prec_u65_L1_corr_prec:', corr_prec_u65_L1_corr_prec)
# print('corr_prec_u65_L1_corr_sett:', corr_prec_u65_L1_corr_sett)
# print('inco_prec_u65_L1_corr_sett:', inco_prec_u65_L1_corr_sett)
# print('inco_prec_u65_L1_inco_sett:', inco_prec_u65_L1_inco_sett)
# print('inco_prec_u65_L1_inco_prec:', inco_prec_u65_L1_inco_prec)

# print('corr_prec_u65_KLD_corr_prec:', corr_prec_u65_KLD_corr_prec)
# print('corr_prec_u65_KLD_corr_sett:', corr_prec_u65_KLD_corr_sett)
# print('inco_prec_u65_KLD_corr_sett:', inco_prec_u65_KLD_corr_sett)
# print('inco_prec_u65_KLD_inco_sett:', inco_prec_u65_KLD_inco_sett)
# print('inco_prec_u65_KLD_inco_prec:', inco_prec_u65_KLD_inco_prec)

# print('corr_prec_u80_SED_corr_prec:', corr_prec_u80_SED_corr_prec)
# print('corr_prec_u80_SED_corr_sett:', corr_prec_u80_SED_corr_sett)
# print('inco_prec_u80_SED_corr_sett:', inco_prec_u80_SED_corr_sett)
# print('inco_prec_u80_SED_inco_sett:', inco_prec_u80_SED_inco_sett)
# print('inco_prec_u80_SED_inco_prec:', inco_prec_u80_SED_inco_prec)

# print('corr_prec_u80_L1_corr_prec:', corr_prec_u80_L1_corr_prec)
# print('corr_prec_u80_L1_corr_sett:', corr_prec_u80_L1_corr_sett)
# print('inco_prec_u80_L1_corr_sett:', inco_prec_u80_L1_corr_sett)
# print('inco_prec_u80_L1_inco_sett:', inco_prec_u80_L1_inco_sett)
# print('inco_prec_u80_L1_inco_prec:', inco_prec_u80_L1_inco_prec)

# print('corr_prec_u80_KLD_corr_prec:', corr_prec_u80_KLD_corr_prec)
# print('corr_prec_u80_KLD_corr_sett:', corr_prec_u80_KLD_corr_sett)
# print('inco_prec_u80_KLD_corr_sett:', inco_prec_u80_KLD_corr_sett)
# print('inco_prec_u80_KLD_inco_sett:', inco_prec_u80_KLD_inco_sett)
# print('inco_prec_u80_KLD_inco_prec:', inco_prec_u80_KLD_inco_prec)

# <<< from precise perspective <<<

# <<< SET-VALUED PREDICTIONS <<<
