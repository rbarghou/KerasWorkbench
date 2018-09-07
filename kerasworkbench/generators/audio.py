import keras
import librosa


class MelDataGenerator(keras.utils.Sequence):
    MEL_FILTERBANK = librosa.filters.mel(sr, 2048)

    def __init__(self,
                 raw_audio,
                 step_size,
                 sample_width,
                 batch_size=32,
                 dim=(32, 32, 32),
                 n_channels=1):
        self.raw_audio = raw_audio
        self.step_size = step_size
        self.sample_width = sample_width
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.samples = [
            self.raw_audio[i: i + sample_width]
            for i in range(0, len(self.raw_audio) - sample_width, step_size)
        ]
        shuffle_map = list(range(len(self.samples)))
        np.random.shuffle(shuffle_map)
        self.shuffle_map = shuffle_map

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        start_sample_idx = index * self.batch_size
        end_sample_idx = start_sample_idx + self.batch_size
        mapped_idxs = self.shuffle_map[start_sample_idx: end_sample_idx]

        Y = [np.abs(librosa.stft(self.samples[idx]))
             for idx in mapped_idxs]

        X = [y.T.dot(self.MEL_FILTERBANK.T).dot(self.MEL_FILTERBANK).T
             for y in Y]

        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=3)
        Y = Y / np.max(Y)
        X = np.array(X)
        X = np.expand_dims(X, axis=3)
        X = X / np.max(X)

        return X, Y

    def on_epoch_end(self):
        shuffle_map = list(range(len(self.samples)))
        np.random.shuffle(shuffle_map)
        self.shuffle_map = shuffle_map
