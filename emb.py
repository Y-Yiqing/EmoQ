import os
import glob
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import csv
from transformers import AutoProcessor, HubertModel  # Use HubertModel instead of Wav2Vec2Model
from tqdm import tqdm
import numpy as np

def find_audio_path(root_dir, filename):
    """
    Recursively search for the file named 'filename' in root_dir and return its full path.
    If not found, return None.
    """
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def extract_frame_embeddings_hubert(audio_path, hubert_processor, hubert_model, device='cpu'):
    """
    Use HuBERT to extract frame-level audio embeddings with shape [time_frames, hidden_dim].
    'time_frames' indicates how many frames the audio is divided into.
    To ensure each sample does not exceed 11 seconds, we truncate the waveform if needed.
    """
    waveform, sr = torchaudio.load(audio_path, format="wav")
    # waveform shape: [channels, samples]
    waveform = waveform.squeeze(0)
    if sr != 16000:
        # Resample to 16kHz if needed
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    # Truncate audio to a maximum of 11 seconds (11 * sr samples)
    max_samples = 11 * sr
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    inputs = hubert_processor(waveform, sampling_rate=sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = hubert_model(input_values)
        # outputs.last_hidden_state has shape: [batch_size, time_frames, hidden_dim]
        # We take the first sample's embeddings, resulting in shape [time_frames, hidden_dim]
        frame_emb = outputs.last_hidden_state[0]
    return frame_emb.cpu()

# Note: The function below is kept for reference but will not be used,
# since we do not process text with RoBERTa in this modified version.
def extract_token_embeddings_roberta(text, tokenizer, roberta_model, device='cpu'):
    """
    Use RoBERTa to extract token-level text embeddings with shape [seq_len, hidden_dim].
    'seq_len' indicates the number of tokens in the text.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = roberta_model(input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state[0]  # shape [seq_len, hidden_dim]
    return token_emb.cpu()

def main():
    # Use CPU (or MPS if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")

    # Initialize HuBERT (Large LS960-ft)
    hubert_name = "facebook/hubert-large-ls960-ft"
    hubert_processor = AutoProcessor.from_pretrained(hubert_name)
    hubert_model = HubertModel.from_pretrained(hubert_name).to(device)
    hubert_model.eval()

    # We are not processing text with RoBERTa now, so we skip its initialization.
    # However, we will still retain the transcription text.

    # Set directories for CSV and audio files
    csv_dir = "/Users/apple/Desktop/URIS/IEMOCAP/iemocap/transcriptions"
    audio_dir = "/Users/apple/Desktop/URIS/IEMOCAP/iemocap/IEMOCAP_full_release"

    # Output file to save the processed data
    output_pt = "/Users/apple/Desktop/URIS/IEMOCAP/iemocap/audio_features.pt"

    # Find all CSV files (e.g., transcriptions_ang.csv, transcriptions_exc.csv, etc.)
    csv_files = glob.glob(os.path.join(csv_dir, "transcriptions_*.csv"))
    if not csv_files:
        print("[Warning] No CSV found matching transcriptions_*.csv")
        return

    # List to store all data for later saving
    all_data_for_pt = []

    # Process each CSV file using tqdm for progress display
    for csv_path in tqdm(csv_files, desc="Processing CSV files", ncols=80):
        # Example: transcriptions_ang.csv => label "ang"
        base = os.path.basename(csv_path)
        name_no_ext = os.path.splitext(base)[0]
        label = name_no_ext.replace("transcriptions_", "")

        with open(csv_path, "r", encoding="utf-8") as fcsv:
            csv_reader = csv.reader(fcsv)
            header = next(csv_reader, None)
            if not header:
                print(f"[Warning] {csv_path} is empty!")
                continue

            for row in csv_reader:
                if len(row) < 2:
                    continue
                filename, transcription = row[0], row[1]

                # Find the full path for the audio file
                audio_path = find_audio_path(audio_dir, filename)
                if not audio_path:
                    print(f"[Warning] Audio file not found for: {filename}")
                    continue

                # Extract frame-level audio embeddings (with truncation to 11 seconds)
                audio_emb = extract_frame_embeddings_hubert(
                    audio_path, hubert_processor, hubert_model, device
                )
                audio_token_count = audio_emb.size(0)  # Number of time frames

                # We no longer process text embeddings.
                # Instead, we keep the transcription text as is.
                # Set text_token_count and text_emb to None (or optionally, leave them out)
                text_token_count = None
                text_emb = None

                # Save information in a dictionary
                item = {
                    "filename": filename,
                    "audio_token_count": audio_token_count,  # Number of frames
                    "audio_emb": audio_emb,   # Shape: [time_frames, hidden_dim]
                    "text": transcription,  # Retain the text content
                    "label": label
                }
                all_data_for_pt.append(item)
                print(f"[Info] Processed {filename} ({label}): {audio_token_count} frames.")

    # Save all collected data to a .pt file
    torch.save(all_data_for_pt, output_pt)
    print(f"[Info] All done. Data saved to {output_pt}, total entries: {len(all_data_for_pt)}.")

if __name__ == "__main__":
    main()

