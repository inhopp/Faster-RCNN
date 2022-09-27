import os
import pandas as pd

train = pd.DataFrame(columns=['fname'])
test = pd.DataFrame(columns=['fname'])

for f_name in os.listdir('./train'):
    if '.jpg' in f_name:
        file_path = 'train/' + f_name
        train = train.append({'fname': file_path}, ignore_index=True)

train.to_csv('train.csv', index=False)

for f_name in os.listdir('./test'):
    if '.jpg' in f_name:
        file_path = 'test/' + f_name
        test = test.append({'fname': file_path}, ignore_index=True)

test.to_csv('test.csv', index=False)


print("########### make csv Ended ###########")