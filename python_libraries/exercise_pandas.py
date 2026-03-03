import pandas as pd
import numpy as np


arr = np.random.randn(4, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

print(df)