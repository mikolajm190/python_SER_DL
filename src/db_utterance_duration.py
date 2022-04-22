import os
import librosa as lb
import pandas as pd

path_to_data = '../data/'
db_list = os.listdir(path_to_data)
duration_list = []

for d in db_list:
    db_max, db_mean, counter = 0, 0, 0
    for root, _, files in os.walk(os.path.join(path_to_data, d)):
        if len(files):
            for filename in files:
                if filename.split('.')[1] == 'wav':
                    y, sr = lb.load(os.path.join(root, filename), sr=44100, res_type='kaiser_fast')
                    y_duration = lb.get_duration(y=y, sr=sr)

                    # max
                    if y_duration > db_max:
                        db_max = y_duration
                    
                    # mean
                    db_mean += y_duration
                    counter += 1

    db_mean /= counter

    duration_list.append([db_max, db_mean])

df = pd.DataFrame(duration_list)
df = pd.concat([pd.DataFrame(db_list), df], axis=1)
df.columns = ['db', 'max', 'mean']
print(df)
