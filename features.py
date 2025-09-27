import torch
import torchaudio
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False
        )
    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform, self.sample_rate

    def compute_mel_spectrogram(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9)  # Adding a small value to avoid log(0)
        return log_mel_spec
