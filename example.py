import torch
from model import ECG_ResNeXt

# Function to process the ECG data before feeding it into the model
def get_inputs(device, ecg_batch, apply="non_zero", signal_crop_len=2560):
    # Process ECG data
    if ecg_batch.shape[1] > ecg_batch.shape[2]:
        ecg_batch = ecg_batch.permute(0, 2, 1)
    B, n_leads, signal_len = ecg_batch.shape

    if apply == "non_zero":
        transformed_ecg = torch.zeros(B, n_leads, signal_crop_len)
        for b in range(B):
            # Infer signal_non_zero_start dynamically for each ECG
            start = torch.nonzero(ecg_batch[b, :, :], as_tuple=False)
            if start.nelement() == 0:
                start = 0
            else:
                start = start[0, 1].item()
            
            end = start + signal_crop_len
            # Adjust start and end if end exceeds signal_len
            if end > signal_len:
                end = signal_len
                start = end - signal_crop_len
                if start < 0:
                    start = 0

            for l in range(n_leads):
                transformed_ecg[b, l, :] = ecg_batch[b, l, start:end]
    else:
        transformed_ecg = ecg_batch

    return transformed_ecg.to(device)

# Initialize model parameters
n_classes = 6
num_blocks = 3
channels = [64, 128, 192, 256]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ECG_ResNeXt model and move it to the device
model = ECG_ResNeXt(n_classes=n_classes, num_blocks=num_blocks, channels=channels).to(device)

# Generate a random ECG signal (batch size of 16, 12 leads, signal length of 4096)
batch_size = 16
signal_len = 4096
n_leads = 12

# Create a random ECG signal
ecg_batch = torch.randn(batch_size, n_leads, signal_len).to(device)

# Process the ECG batch using the get_inputs function
processed_ecg = get_inputs(device, ecg_batch, signal_crop_len=2560)

# Perform a forward pass
logits = model(processed_ecg)

# Print the input shape and output shape
print(f"Input shape: {processed_ecg.shape}")
print(f"Output shape: {logits.shape}")
