import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset, collate_fn
from model import SpeechModel
from decoder import CTCBeamSearchDecoder
import editdistance

def cer(pred_text, target_text):
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0
    return editdistance.eval(pred_text, target_text) / len(target_text)

def evaluate(model, loader, criterion, dataset, decoder, device):
    model.eval()
    total_loss, total_cer, num_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for features, transcripts, feat_lens, trans_lens in loader:
            features, transcripts = features.to(device), transcripts.to(device)

            # Forward
            log_probs = model(features).log_softmax(dim=-1)  # (T, B, vocab)

            # CTC Loss
            loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
            total_loss += loss.item()

            print("Log_probs shape:", log_probs.shape)  # (T, batch, vocab_size)
            print("Log_probs sample:", log_probs[:5, 0, :5])

            # Decode
            preds = decoder.decode(log_probs)

            start = 0
            for i, length in enumerate(trans_lens):
                target_seq = transcripts[start:start+length].cpu().numpy().tolist()
                start += length
                target_text = "".join([dataset.idx2char[c] for c in target_seq])
                pred_text = "".join([dataset.idx2char[c] for c in preds[i]]) if preds[i] else ""

                        
                if i < 3:
                    print("Pred:", pred_text)
                    print("Target:", target_text)

                total_cer += cer(pred_text, target_text)
                num_samples += 1

    avg_loss = total_loss / len(loader)
    avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
    return avg_loss, avg_cer


def train_ctc(num_epochs=50, batch_size=8, lr=5e-4, hidden_dim=512, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    dataset = Dataset()

    
    
    train_loader = DataLoader(dataset.get_all_data(), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset.get_validation_data(), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    features, transcripts, feat_lens, trans_lens = next(iter(train_loader))
    print("Features:", features.shape)      # (max_time, batch, feat_dim)
    print("Transcripts:", transcripts.shape) # (sum of label lengths,)
    print("Feature lengths:", feat_lens)
    print("Transcript lengths:", trans_lens)

    # Model
    input_dim = train_loader.dataset[0][0].shape[1]  # feature dimension
    output_dim = len(dataset.char2idx) + 1  # + blank
    model = SpeechModel(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    decoder = CTCBeamSearchDecoder(idx2char=dataset.idx2char, blank=0, beam_width=10)
    best_cer = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for features, transcripts, feat_lens, trans_lens in train_loader:
            features, transcripts = features.to(device), transcripts.to(device)

            # Check that feature lengths >= transcript lengths
            for f_len, t_len in zip(feat_lens, trans_lens):
                if f_len < t_len:
                    print(f"Warning: feature length {f_len} < transcript length {t_len}")

            optimizer.zero_grad()
            log_probs = model(features).log_softmax(dim=-1)  # (T, B, vocab)
            loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, avg_val_cer = evaluate(model, val_loader, criterion, dataset, decoder, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val CER: {avg_val_cer:.4f}")

        if avg_val_cer < best_cer:
            best_cer = avg_val_cer
            torch.save(model.state_dict(), "best_ctc_model.pth")
            print(f"Saved new best model at epoch {epoch+1} with CER {best_cer:.4f}")

    return model


if __name__ == "__main__":
    train_ctc()
