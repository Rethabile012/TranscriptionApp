import os
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from features import AudioProcessor  

class Dataset:
    def __init__(self, 
                 audio_root="TrainingSet/audio", 
                 transcript_root="TrainingSet/transcripts",
                 vocab="abcdefghijklmnopqrstuvwxyz '",
                 sample_rate=16000):
        self.audio_root = audio_root
        self.transcript_root = transcript_root
        self.processor = AudioProcessor(sample_rate=sample_rate)

        # Build char vocab
        self.char2idx = {c: i+1 for i, c in enumerate(vocab)}  # +1 so blank=0
        self.idx2char = {i+1: c for i, c in enumerate(vocab)}
        self.blank_id = 0

    def clean_text(self, text: str) -> str:
        
        text = text.lower()

        # remove tokens like i_letter, a_letter, etc.
        text = re.sub(r"\b[a-z]_letter\b", "", text)

        # keep only vocab chars (a-z, space, apostrophe)
        text = re.sub(r"[^a-z' ]", "", text)

        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def load_text(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                return self.clean_text(raw)
        except Exception as e:
            print(f"Failed to load transcript {filepath}: {e}")
            return ""

    def text_to_int(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def load_pairs(self, audio_root, transcript_root):
        dataset = []
        for dirpath, _, filenames in os.walk(audio_root):
            relative_folder = os.path.relpath(dirpath, audio_root)
            transcript_dirpath = os.path.join(transcript_root, relative_folder)

            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    audio_path = os.path.join(dirpath, filename)
                    transcript_path = os.path.join(
                        transcript_dirpath, filename.replace(".wav", ".txt")
                    )

                    if not os.path.exists(transcript_path):
                        print(f"Warning: Missing transcript for {audio_path}")
                        continue

                    try:
                        # Load audio
                        waveform, sr = self.processor.load_audio(audio_path)

                        # Ensure mono
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        waveform = waveform.squeeze(0)

                        # Features
                        mel_spec = self.processor.compute_mel_spectrogram(waveform)

                        # Transcript → tokens
                        transcript_text = self.load_text(transcript_path)
                        transcript_ints = self.text_to_int(transcript_text)

                        if len(transcript_ints) == 0:
                            print(f"Skipping empty transcript after cleaning: {transcript_path}")
                            continue

                        dataset.append((mel_spec.detach().clone().transpose(0, 1),  # (T, F)
                                        torch.tensor(transcript_ints, dtype=torch.long)
                                        ))
                    except Exception as e:
                        print(f"Failed to process {audio_path}: {e}")
        return dataset

    def get_all_data(self):
        return self.load_pairs(self.audio_root, self.transcript_root)

    def get_validation_data(self, audio_root="ValidationSet/audio", transcript_root="ValidationSet/transcripts"):
        return self.load_pairs(audio_root, transcript_root)

    def get_test_data(self, audio_root="TestSet/audio", transcript_root="TestSet/transcripts"):
        return self.load_pairs(audio_root, transcript_root)


