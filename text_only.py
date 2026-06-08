import pandas as pd
import os

""" Load your original dataset
    Create a text-only version by emptying the image column
    Save it as a new file in the SAME directory
"""
df = pd.read_csv('/srv/essa-lab/flash3/cnimo3/work/reproduction/afrimedqa-vlmeval/test_files/YORUBA_TEST__Sheet1.tsv', sep='\t')


df['image'] = '' 


df.to_csv('/srv/essa-lab/flash3/cnimo3/work/reproduction/afrimedqa-vlmeval/test_files/YORUBA_TEST__Sheet1_TextOnly.tsv', sep='\t', index=False)