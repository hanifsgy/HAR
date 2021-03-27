from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from scipy import stats
from IPython.display import display, HTML
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# The WISDM dataset contains six different labels
# (Downstairs, Jogging, Sitting, Standing, Upstairs, Walking)

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
# Default seaborn look and feels
sns.set()
plt.style.use('ggplot')
print('keras version', keras.__version__)
# Assing the labels for throughout the program
LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next
STEP_DISTANCE = 40

def read_data(file_path):
    column_names = [
        'user-id',
        'activity',
        'timestamp',
        'x-axis',
        'y-axis',
        'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # ... and then this column must be transformed to float explicity
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # this is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

def show_basic_dataframe_info(dataframe):
    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

df = read_data('WISDM_ar_v1.1_raw.txt')

# describe data
show_basic_dataframe_info(df)
df.head(20)

df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()
# Better understand how the recordings are spread across the different
# users who participated in the study
df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()
