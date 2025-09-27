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
        for batch_idx, (features, transcripts, feat_lens, trans_lens) in enumerate(loader):
            # Get actual batch size (handles last batch with fewer samples)
            actual_batch_size = features.shape[0]
            
            # Ensure length tensors match actual batch size
            if feat_lens.shape[0] != actual_batch_size:
                print(f"Warning: feat_lens size {feat_lens.shape[0]} != batch size {actual_batch_size}")
                continue
            if trans_lens.shape[0] != actual_batch_size:
                print(f"Warning: trans_lens size {trans_lens.shape[0]} != batch size {actual_batch_size}")
                continue
                
            features, transcripts = features.to(device), transcripts.to(device)
            feat_lens, trans_lens = feat_lens.to(device), trans_lens.to(device)

            try:
                # Forward pass
                logits = model(features)  # (B, T, C)
                log_probs = logits.log_softmax(dim=-1)  # (B, T, C)
                log_probs = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC

                # Validate shapes
                if log_probs.shape[1] != actual_batch_size:
                    print(f"Warning: log_probs batch dim {log_probs.shape[1]} != actual batch size {actual_batch_size}")
                    continue

                # CTC Loss
                loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
                total_loss += loss.item()

                if batch_idx == 0:  # Debug first batch only
                    print("Log_probs shape:", log_probs.shape)
                    print("Log_probs sample:", log_probs[:5, 0, :5])

                # Debug decoder output
                if batch_idx == 0:
                    print("\n=== DETAILED DECODER DEBUG ===")
                    
                    # Check raw predictions
                    greedy_preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (batch, time)
                    print(f"Raw greedy predictions shape: {greedy_preds.shape}")
                    print(f"First sample greedy (first 50): {greedy_preds[0][:50].tolist()}")
                    
                    # Check unique values in predictions
                    unique_vals = torch.unique(greedy_preds[0])
                    print(f"Unique prediction indices: {unique_vals.tolist()}")
                    
                    # Check log probabilities
                    print(f"Log prob stats - min: {log_probs.min():.3f}, max: {log_probs.max():.3f}")
                    print(f"Blank token (idx 0) avg prob: {log_probs[:, 0, 0].mean():.3f}")
                    print(f"Non-blank avg prob: {log_probs[:, 0, 1:].mean():.3f}")
                    
                    print("==============================\n")

                # Decode predictions
                preds = decoder.decode(log_probs)

                # Debug beam search results
                if batch_idx == 0:
                    print(f"Beam search returned {len(preds)} predictions")
                    for i in range(min(3, len(preds))):
                        if preds[i]:
                            print(f"Beam search sample {i}: {preds[i][:20]}")
                        else:
                            print(f"Beam search sample {i}: EMPTY")

                # Fallback to greedy if beam search fails
                if all(not pred for pred in preds):  # If all predictions are empty
                    if batch_idx == 0:
                        print("Beam search failed, trying greedy...")
                    greedy_preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (batch, time)
                    
                    # Simple greedy decode (remove consecutive duplicates and blanks)
                    preds = []
                    for b in range(greedy_preds.shape[0]):
                        sequence = greedy_preds[b].cpu().numpy()
                        # Remove blanks (0) and consecutive duplicates
                        decoded = []
                        prev = -1
                        for char in sequence:
                            if char != 0 and char != prev:  # Not blank and not duplicate
                                decoded.append(char)
                            prev = char
                        preds.append(decoded)

                # Calculate CER
                start = 0
                for i, length in enumerate(trans_lens):
                    target_seq = transcripts[start:start+length].cpu().numpy().tolist()
                    start += length
                    target_text = "".join([dataset.idx2char[c] for c in target_seq])
                    
                    # Safe character mapping
                    if preds[i]:
                        pred_chars = []
                        for c in preds[i]:
                            if c in dataset.idx2char:
                                pred_chars.append(dataset.idx2char[c])
                            else:
                                print(f"Warning: Unknown character index {c}, skipping")
                        pred_text = "".join(pred_chars)
                    else:
                        pred_text = ""
                    
                    if batch_idx == 0 and i < 3:  # Show first 3 examples of first batch
                        print(f"Sample {i}:")
                        print("  Pred:", repr(pred_text[:50]))
                        print("  Target:", repr(target_text[:50]))

                    total_cer += cer(pred_text, target_text)
                    num_samples += 1

            except RuntimeError as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                print(f"Shapes - features: {features.shape}, feat_lens: {feat_lens.shape}, trans_lens: {trans_lens.shape}")
                continue

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    avg_cer = total_cer / num_samples if num_samples > 0 else 1.0
    return avg_loss, avg_cer


def collate_fn(batch):
    features, transcripts = zip(*batch)

    # Pad features to (B, T, F) format - BATCH FIRST
    feature_lengths = [f.shape[0] for f in features]
    max_len = max(feature_lengths)
    feat_dim = features[0].shape[1]
    features_padded = torch.zeros(len(batch), max_len, feat_dim)  # (B, T, F)
    for i, f in enumerate(features):
        features_padded[i, :f.shape[0], :] = f

    # Concatenate transcripts
    transcript_lengths = [len(t) for t in transcripts]
    transcripts_concat = torch.cat(transcripts)

    # Convert to tensors with proper dtypes
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)

    return features_padded, transcripts_concat, feature_lengths, transcript_lengths


def train_ctc(num_epochs=50, batch_size=8, lr=5e-5, hidden_dim=512, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = Dataset()
    
    # Debug character vocabulary
    print("\n=== CHARACTER VOCABULARY DEBUG ===")
    print("Character vocabulary:")
    for i, char in enumerate(dataset.idx2char):
        print(f"{i}: '{char}' (ord: {ord(char) if len(char) == 1 else 'N/A'})")
    print(f"Total vocab size: {len(dataset.idx2char)}")
    print(f"Expected output dim: {len(dataset.char2idx) + 1}")  # +1 for blank
    print("=====================================\n")
    
    train_loader = DataLoader(
        dataset.get_all_data(),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True  # Ensure consistent batch sizes
    )
    val_loader = DataLoader(
        dataset.get_validation_data(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True  # Ensure consistent batch sizes
    )

    # Peek one batch for sanity check
    features, transcripts, feat_lens, trans_lens = next(iter(train_loader))
    print("Features:", features.shape)       # (batch, max_time, feat_dim)
    print("Transcripts:", transcripts.shape) # (sum of label lengths,)
    print("Feature lengths:", feat_lens.shape, feat_lens[:5])
    print("Transcript lengths:", trans_lens.shape, trans_lens[:5])

    # Model
    input_dim = features.shape[2]  # feature dimension
    output_dim = len(dataset.char2idx) + 1  # +1 for blank
    print(f"Model dims: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
    
    model = SpeechModel(input_dim, hidden_dim, output_dim).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    decoder = CTCBeamSearchDecoder(
        idx2char=dataset.idx2char,
        blank=0,
        beam_width=10
    )

    best_cer = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, transcripts, feat_lens, trans_lens) in enumerate(train_loader):
            features, transcripts = features.to(device), transcripts.to(device)
            feat_lens, trans_lens = feat_lens.to(device), trans_lens.to(device)

            # Validate batch
            actual_batch_size = features.shape[0]
            if feat_lens.shape[0] != actual_batch_size or trans_lens.shape[0] != actual_batch_size:
                print(f"Skipping batch {batch_idx}: inconsistent sizes")
                continue

            # Check that feature lengths >= transcript lengths
            invalid_samples = (feat_lens < trans_lens).any()
            if invalid_samples:
                print(f"Warning: Some feature lengths < transcript lengths in batch {batch_idx}")

            optimizer.zero_grad()

            try:
                # Forward pass
                logits = model(features)  # (B, T, C)
                log_probs = logits.log_softmax(dim=-1)  # (B, T, C)
                log_probs = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC

                # Debug first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"Raw logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                    print(f"Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
                    print("Log probs shape:", log_probs.shape)

                # CTC Loss
                loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss in batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
                # Check for gradient explosion
                if grad_norm > 10.0:
                    print(f"Large gradient norm: {grad_norm:.3f}")
                
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Print progress every 50 batches
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Grad norm: {grad_norm:.3f}")

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Validation
        print("Running validation...")
        avg_val_loss, avg_val_cer = evaluate(model, val_loader, criterion, dataset, decoder, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val CER: {avg_val_cer:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Learning rate scheduling
        scheduler.step(avg_val_cer)

        # Save best model
        if avg_val_cer < best_cer:
            best_cer = avg_val_cer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, "best_ctc_model.pth")
            print(f"Saved new best model with CER {best_cer:.4f}")

        # Early stopping if CER stops improving
        if avg_val_cer > best_cer * 2 and epoch > 10:
            print("Early stopping: validation CER not improving")
            break

    print(f"Training completed. Best CER: {best_cer:.4f}")
    return model



if __name__ == "__main__":
    train_ctc()