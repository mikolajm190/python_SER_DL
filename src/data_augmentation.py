#!/home/judehey/miniconda3/envs/DL_env/bin/python

import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf

def load_data(path):
    emotion, file_path, database = [], [], []
    
    # iterate over audio files extracting emotion label, file path and speaker info
    for root, _, files in os.walk(path):
        if len(files): # extract info only if files are found
            for filename in files:
                temp_path_split = os.path.dirname(root).split('/')
                # extract emotion label specific to every database
                if temp_path_split[2] == 'emodb':
                    emotion.append(filename[5])
                elif temp_path_split[2] == 'emovo':
                    emotion.append(filename.split('-')[0])
                elif temp_path_split[2] == 'ravdess':
                    emotion.append(filename.split('-')[2])
                elif temp_path_split[2] == 'tess':
                    emotion.append(filename.split('_')[2].split('.')[0])
                elif temp_path_split[2] == 'shemo':
                    emotion.append(filename[3])
                elif temp_path_split[2] == 'aesdd':
                    emotion.append(os.path.basename(root))
                file_path.append(os.path.join(root, filename))
                database.append(temp_path_split[2])

    # prepare dataframe
    audio_df = pd.DataFrame(emotion)
    audio_df = pd.concat([pd.DataFrame(file_path), pd.DataFrame(database), audio_df], axis=1)
    audio_df.columns = ['path', 'database', 'emotion']
    
    return audio_df


# fear -> anxiety (done), boredom -> neutral (done), calm -> neutral (just a consideration)
def translate_labels(df):
    # replace label inidication with full name for each database
    df.loc[df['database'] == 'emodb'] = df.loc[df['database'] == 'emodb'].replace({'W': 'anger', 'L': 'neutral',
                                                                               'E': 'disgust', 'A': 'anxiety',
                                                                               'F': 'happiness', 'T': 'sadness',
                                                                               'N': 'neutral'})
    df.loc[df['database'] == 'emovo'] = df.loc[df['database'] == 'emovo'].replace({'neu': 'neutral', 'dis': 'disgust',
                                                                               'gio': 'happiness','rab': 'anger',
                                                                               'sor': 'surprise', 'tri': 'sadness',
                                                                               'pau': 'anxiety'})
    df.loc[df['database'] == 'ravdess'] = df.loc[df['database'] == 'ravdess'].replace({'01': 'neutral', '02': 'calm',
                                                                                   '03': 'happiness', '04': 'sadness',
                                                                                   '05': 'anger', '06': 'anxiety',
                                                                                   '07': 'disgust', '08': 'surprise'})
    df.loc[df['database'] == 'shemo'] = df.loc[df['database'] == 'shemo'].replace({'A': 'anger', 'F': 'anxiety',
                                                                               'H': 'happiness', 'N': 'neutral',
                                                                               'S': 'sadness', 'W': 'surprise'})
    df.loc[df['database'] == 'tess'] = df.loc[df['database'] == 'tess'].replace({'sad': 'sadness', 'angry': 'anger',
                                                                                 'ps': 'surprise', 'fear': 'anxiety',
                                                                                  'happy': 'happiness', 'surprised': 'surprise'})
    df.loc[df['database'] == 'aesdd'] = df.loc[df['database'] == 'aesdd'].replace({'fear': 'anxiety'})
    
    return df


def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_audio(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(y=data, rate=speed_factor)


def augment(y, sampling_rate, aug_options):
    # if none augmentation option is specified choose random
    if not(aug_options[0]) and not(aug_options[1]) and not(aug_options[2]) and not(aug_options[3]):
        aug_index = rng.integers(low=0, high=3, endpoint=True)
        aug_options[aug_index] = True
        
    augmented_audio = y
        
    if aug_options[0]: # pitch change
        pitch_factor = rng.random() - 0.5
        augmented_audio = change_pitch(augmented_audio, sampling_rate, pitch_factor)
    
    if aug_options[1]: # change speed
        speed_factor = 0.2 * rng.random() + 0.9
        augmented_audio = change_speed(augmented_audio, speed_factor)
    
    if aug_options[2]: # add noise
        noise_factor = 0.009 * rng.random() + 0.001
        augmented_audio = add_noise(augmented_audio, noise_factor)
    
    if aug_options[3]: # shift audio
        shift_max = 0.1 * librosa.get_duration(y=y, sr=sampling_rate)
        augmented_audio = shift_audio(augmented_audio, sampling_rate, shift_max, 'both')
    
    return augmented_audio


def augment_utterance(utterance, sampling_rate):
    # pick random augmentation modes
    augmentation_modes = rng.integers(low=0, high=1, size=4, endpoint=True)
    
    # augment based on random values
    augmented_utterance = augment(utterance, sampling_rate, augmentation_modes)
    
    return augmented_utterance


def augment_audio_df(df):
    # prepare directory for all data that will be augmented
    # for each emotion separately (without anger, which is the greatest class)
    for emo in df['emotion'].unique():
        if emo != 'anger':
            os.makedirs(os.path.join(path_to_data, 'augmented', emo), exist_ok=True)
            
    # augment data in one of four ways:
    # change pitch, speed, shift audio and add noise
    # do it until class utterances reach anger class count
    for emo in df['emotion'].unique():
        utterances_diff_count = df['emotion'].value_counts()['anger'] - df['emotion'].value_counts()[emo]
        for n in range(utterances_diff_count):
            # pick random utterance with corresponding emotion label
            random_index = np.random.choice(df[df['emotion'] == emo].index.values)
            
            # get path to utterance
            file_path = df.iloc[random_index].values[0]
            
            # load file
            utterance, sampling_rate = librosa.load(file_path, sr=44100)
            
            # alter a file
            augmented_utterance = augment_utterance(utterance, sampling_rate)
            
            # save new file
            new_file_name = f"{emo}_{n}_{os.path.basename(file_path).split('.')[0]}.wav"
            sf.write(file=os.path.join(path_to_data, 'augmented', emo, new_file_name),
                     data=augmented_utterance, samplerate=44100)



def main():
    path_to_data='../data/'
    imgs_path='../imgs/cross_corpus_augmented/'
    models_path='../models/cross_corpus_augmented/'

    # random numbers generator init
    rng = np.random.default_rng(seed=44)

    df = load_data(path_to_data)
    df = translate_labels(df)

    augment_audio_df(df)


if __name__ == '__main__':
    main()