import matplotlib.pyplot as plt
import pandas as pd

bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'sample']
df = pd.read_csv('/home/ubuntu/codebase/tf_binding/data/hg38/tf_seq_data/pos/AR.bed', names = bed_cols)

# Plot histogram of signal values
df.hist(column='score', bins=100)
plt.savefig('AR_signal_dist.png')
