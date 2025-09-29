import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import SpeechModel
from decoder import CTCBeamSearchDecoder
import editdistance

# -------------------------
# Utilities
# -------------------------
def cer(pred_text, target_text):
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0
    return editdistance.eval(pred_text, target_text) / len(target_text)

# -------------------------
# Evaluation (unchanged mostly, with a little more debug)
# -------------------------
def evaluate(model, loader, criterion, dataset, decoder, device, logit_scale=None, idx2char=None):
    model.eval()
    total_loss, total_cer, num_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for batch_idx, (features, transcripts, feat_lens, trans_lens) in enumerate(loader):
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

                # Apply learnable logit scale if provided (makes logits less flat)
                if logit_scale is not None:
                    logits = logits * logit_scale

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

                # Debug decoder / greedy output for the first batch
                if batch_idx == 0:
                    print("\n=== DETAILED DECODER DEBUG ===")
                    greedy_preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (batch, time)
                    print(f"Raw greedy predictions shape: {greedy_preds.shape}")
                    print(f"First sample greedy (first 50): {greedy_preds[0][:50].tolist()}")
                    unique_vals = torch.unique(greedy_preds[0])
                    print(f"Unique prediction indices: {unique_vals.tolist()}")
                    print(f"Log prob stats - min: {log_probs.min():.3f}, max: {log_probs.max():.3f}")
                    # Average across time for sample 0
                    print(f"Blank token (idx 0) avg prob: {log_probs[:, 0, 0].mean():.3f}")
                    print(f"Non-blank avg prob: {log_probs[:, 0, 1:].mean():.3f}")

                    # Print top-k classes at some random timesteps for sample 0 (helps debugging)
                    for t in [0, max(0, greedy_preds.shape[1]//4), max(0, greedy_preds.shape[1]//2)]:
                        topk = torch.topk(log_probs[t, 0, :], k=min(5, log_probs.shape[-1]))
                        print(f"Time {t} topk indices: {topk.indices.tolist()}, values: {[float(x) for x in topk.values.tolist()]}")
                    print("==============================\n")

                # Decode predictions
                preds = decoder.decode(log_probs)

                # Debug beam search results
                if batch_idx == 0:
                    print(f"Beam search returned {len(preds)} predictions")
                    for i in range(min(3, len(preds))):
                        if preds[i]:
                            print(f"Beam search sample {i}: {preds[i][:40]}")
                        else:
                            print(f"Beam search sample {i}: EMPTY")

                # Fallback to greedy if beam search fails (empty predictions)
                if all(not pred for pred in preds):
                    if batch_idx == 0:
                        print("Beam search failed, trying greedy fallback...")
                    greedy_preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (batch, time)

                    preds = []
                    for b in range(greedy_preds.shape[0]):
                        sequence = greedy_preds[b].cpu().numpy()
                        decoded = []
                        prev = -1
                        for char in sequence:
                            if char != 0 and char != prev:  # not blank and not duplicate
                                decoded.append(char)
                            prev = char
                        preds.append(decoded)

                # Calculate CER
                start = 0
                for i, length in enumerate(trans_lens):
                    target_seq = transcripts[start:start+length].cpu().numpy().tolist()
                    start += length

                    # Build target text from idx2char mapping passed into evaluate
                    if idx2char is None:
                        # fallback to dataset mapping if not provided
                        mapping = dataset.idx2char
                    else:
                        mapping = idx2char

                    target_text = "".join([mapping[c] for c in target_seq if c in mapping])

                    # preds[i] may be either:
                    #  - a string (if a decoder returned a string) OR
                    #  - a list of integer token indices (the unified format)
                    if isinstance(preds[i], str):
                        pred_text = preds[i]
                    else:
                        # assume list of ints
                        pred_chars = []
                        for c in preds[i]:
                            if c in mapping and mapping[c] != "<BLANK>":
                                pred_chars.append(mapping[c])
                        pred_text = "".join(pred_chars)

                    if batch_idx == 0 and i < 3:
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

# -------------------------
# Collate function (kept your implementation)
# -------------------------
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

# -------------------------
# Training entrypoint (main edits here)
# -------------------------
def train_ctc(num_epochs=50, batch_size=8, lr=1e-3, hidden_dim=512, device=None):
    """
    Changes from original:
    - Default lr increased to 1e-3 (try from-scratch training).
    - Added learnable `logit_scale` parameter initialized to 5.0 and multiplied into logits
      before softmax; this is registered in the optimizer so it is trained.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = Dataset()

    # Debug character vocabulary
    print("\n=== CHARACTER VOCABULARY DEBUG ===")
    print("Character vocabulary:")
    for i, char in dataset.idx2char.items():
        print(f"{i}: '{char}' (ord: {ord(char) if isinstance(char, str) and len(char) == 1 else 'N/A'})")
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

    # Criterion & optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Learnable logit scale to widen logits before softmax (helps avoid flat logits)
    logit_scale = nn.Parameter(torch.tensor(5.0, dtype=torch.float32, device=device), requires_grad=True)
    # Set up optimizer to include the logit_scale parameter
    optimizer = torch.optim.Adam(list(model.parameters()) + [logit_scale], lr=lr)

    # Learning rate scheduler (monitor val CER; keep original behavior)
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

            # Check that feature lengths >= transcript lengths (simple sanity)
            if (feat_lens < trans_lens).any():
                print(f"Warning: Some feature lengths < transcript lengths in batch {batch_idx}")

            optimizer.zero_grad()

            try:
                # Forward pass
                logits = model(features)  # (B, T, C)

                # Apply learnable logit scale (this will be trained)
                logits = logits * logit_scale

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

                # Gradient clipping and monitoring
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if grad_norm > 10.0:
                    print(f"Large gradient norm: {grad_norm:.3f}")

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Print progress every 50 batches
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Grad norm: {grad_norm:.3f}, logit_scale: {float(logit_scale):.3f}")

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Validation
        print("Running validation...")
        avg_val_loss, avg_val_cer = evaluate(
              model, val_loader, criterion, dataset, decoder, device,
              logit_scale=logit_scale, idx2char=dataset.idx2char
          )

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val CER: {avg_val_cer:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"logit_scale: {float(logit_scale):.3f}")

        # Learning rate scheduling (using CER here, same as your original script)
        scheduler.step(avg_val_cer)

        # Save best model
        if avg_val_cer < best_cer:
            best_cer = avg_val_cer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'logit_scale': float(logit_scale.detach().cpu().numpy()),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, "best_ctc_model.pth")
            print(f"Saved new best model with CER {best_cer:.4f}")

        # Early stopping if CER stops improving drastically
        if avg_val_cer > best_cer * 2 and epoch > 10:
            print("Early stopping: validation CER not improving")
            break

    print(f"Training completed. Best CER: {best_cer:.4f}")
    return model

if __name__ == "__main__":
    train_ctc()
