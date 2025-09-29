import torch
from dataset import Dataset, collate_fn
from model import SpeechModel
from decoder import CTCBeamSearchDecoder

import editdistance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpeechModel(input_dim=80, hidden_dim=512, output_dim=29)
model.load_state_dict(torch.load("best_ctc_model.pth", map_location=device))
model.to(device)
model.eval()


from torch.utils.data import DataLoader
test_dataset = Dataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

 
dataset = Dataset()

lm_path = "4gram.arpa"
decoder = CTCBeamSearchDecoder(dataset.idx2char, lm=lm_path)




def cer(pred_text, target_text):
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0
    return editdistance.eval(pred_text, target_text) / len(target_text)

total_cer, count = 0, 0

with torch.no_grad():
    for features, feature_lengths, transcripts, transcript_lengths in test_loader:
        features, feature_lengths = features.to(device), feature_lengths.to(device)
        
        # Forward
        log_probs, output_lengths = model(features, feature_lengths)
        log_probs = log_probs.cpu().numpy()  # shape [T, V]
        
        # Decode with LM
        pred_text = decoder.decode(log_probs[0])  # since batch=1
        target_text = "".join(transcripts[0])     # adjust to your dataset format
        
        total_cer += cer(pred_text, target_text)
        count += 1

print(f"Test CER with LM: {total_cer/count:.4f}")
