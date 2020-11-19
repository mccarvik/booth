
import pdb
import numpy as np
import pandas as pd

df = pd.read_excel("PS3_Q2.xls")[1:].set_index("Quarter")
pdb.set_trace()
print(df.mean())
print(df.std())
print(df.cov())
print(df.corr())