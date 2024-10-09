import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import datetime
import math
from sklearn.decomposition import FastICA
from scipy.signal import istft
#import sounddevice as sd
from scipy.ndimage import zoom
import scipy.signal
import soundfile as sf
from scipy.signal import spectrogram
import itertools


def post_process_audio(audio, sample_rate=44100):
    # Normalize the audio to the range [-1, 1]
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

    # Apply a low-pass filter to remove high-frequency noise
    nyquist = 0.5 * sample_rate
    cutoff = 8000  # Cutoff frequency in Hz
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(1, normal_cutoff, btype='low', analog=False)
    audio = scipy.signal.filtfilt(b, a, audio)

    return audio

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run ICA
def run_ica(emg_data, num_components):
    """
    Apply ICA to 16-channel EMG data and return the independent components.
    
    Args:
        emg_data: A 2D numpy array of shape (num_samples, num_channels) where each column is a channel.
        num_components: The number of independent components to extract (typically the same as the number of channels).
        
    Returns:
        independent_components: The separated independent components after applying ICA.
    """
    # Initialize FastICA object
    ica = FastICA(n_components=num_components, max_iter=10000 , random_state=42, tol=0.1)

    # Fit the ICA model to the EMG data and recover the independent components
    independent_components = ica.fit_transform(emg_data)
    
    return independent_components, ica

# Plot the original EMG signals and the extracted independent components
def plot_signals(original_signals, ica_signals):
    """
    Plot the original EMG signals and the separated independent components.
    
    Args:
        original_signals: A 2D numpy array with original EMG data (shape: [num_samples, num_channels]).
        ica_signals: A 2D numpy array with the ICA components (shape: [num_samples, num_components]).
    """
    
    num_channels = original_signals.shape[0]
    num_samples = original_signals.shape[1]
    time = np.arange(num_samples)

    # Create a figure with 2 subplots (one for original signals, one for ICA signals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))  # (2 rows, 1 column) for vertical stacking

    # Plot original signals on the first subplot
    offset = 500
    for i in range(num_channels):
        ax1.plot(time, original_signals[num_channels - 1 - i] + i * offset, label=f'Channel {i + 1}')
    ax1.set_title('Original EMG Signals')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')

    # Plot ICA signals on the second subplot
    offset = 10
    for i in range(num_channels):
        ax2.plot(time, ica_signals[num_channels - 1 - i] + i * offset, label=f'Channel {i + 1}')
    ax2.set_title('ICA Transformed Signals')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right')

    # Display both plots in the same window
    plt.tight_layout()  # Adjust the layout to prevent overlap
    #plt.show()
    plt.savefig(os.path.join(run_dir, 'ica_plots.png'))  # Save the figure to the specified path


def save_model(model, file_path):
    """Save the model to the specified file path."""
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

def save_spectrogram(spectrogram, file_path, title):
    """Save the spectrogram as an image."""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=22050, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory

def process_labels(labels, sr=44100, n_fft=2048, hop_length=256, n_mels=256, fmin=0, fmax=8000):
    processed_labels = []
    for label in labels:
        try:
            audio = label
            # Convert to numpy array if it's not already
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            # Ensure audio is 1D
            audio = np.mean(audio, axis=1)
            # Generate mel spectrogram
            window = 'blackman'  # or 'hamming', 'blackman', etc.
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, 
                                                     sr=sr, 
                                                     n_fft=n_fft, 
                                                     hop_length=hop_length, 
                                                     n_mels=n_mels,
                                                     fmin=fmin,
                                                     fmax=fmax,
                                                     window=window)

            # Resize the spectrogram to (128, 256)
            if mel_spectrogram.shape[1] != 256:
                # Use interpolation to resize to (256, 256)
                mel_spectrogram = zoom(mel_spectrogram, (256 / mel_spectrogram.shape[0], 256 / mel_spectrogram.shape[1]), order=1)  # Linear interpolation
            
            #Convert Mel spectrogram back to linear spectrogram
            #linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)
            #Reconstruct waveform from linear spectrogram
            #reconstructed_waveform = librosa.istft(linear_spectrogram, hop_length=hop_length)

            #log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max) #this is the normal mel spectrogram but I think it is messing with the reconstruction of the waveform
            #log_mel_spectrogram = mel_spectrogram
            
            processed_labels.append(mel_spectrogram)

            
        except Exception as e:
            print(f"Error processing label: {e}")
            print(f"Label type: {type(label)}, Shape: {np.array(label).shape}")
    return np.array(processed_labels)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))  # Learnable parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the positional encodings
        nn.init.normal_(self.encoding, mean=0, std=0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].unsqueeze(0)  # Add positional encoding
    
# Patch Embedding: Embeds the 1D input into a lower-dimensional representation
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=14, patch_size=50, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        x = self.proj(x)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        x = self.norm(x)
        return x

# Transformer Encoder Block
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.attention(x, x, x)  # [seq_length, batch_size, embed_dim]
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# Full Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, in_channels=14, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4, patch_size=50):
        super(TransformerModel, self).__init__()
        self.d_model = embed_dim  # Store d_model as an instance variable
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        
        # Initialize the learnable positional encoding
        self.positional_encoding = LearnablePositionalEncoding(d_model=embed_dim, max_len=5000)

        # Transformer Encoder
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        
        # Linear projection to match spectrogram size
        self.fc = nn.Linear(embed_dim,256 * 256)  # Adjusted for (256, 256)
        self.embed_dim = embed_dim

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for linear layers
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Kaiming initialization for Conv1d
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        x = x.transpose(0, 1)  # Transformer expects [seq_length, batch_size, embed_dim]

        # Add positional encoding
        x = self.positional_encoding(x)  # Add positional encoding

        # Pass through Transformer Encoder
        for layer in self.transformer:
            x = layer(x)

        x = x.transpose(0, 1)  # [batch_size, num_patches, embed_dim]
        
        # Average pooling to get a single representation
        x = torch.mean(x, dim=1)  # [batch_size, embed_dim]

        # Linear projection to spectrogram size
        x = self.fc(x)  # [batch_size, 256 * 256]
        x = x.view(-1, 256, 256)  # Reshape to [batch_size, 256, 256]

        return x

# Function to display two spectrograms side by side
def save_comparison_spectrograms(true_spectrogram, pred_spectrogram, loss, file_path):
    """Save true and predicted spectrograms side by side."""
    plt.figure(figsize=(12, 6))


    # Compute the min and max values across both spectrograms to set the same color scale
    vmin = min(true_spectrogram.min(), pred_spectrogram.min())
    vmax = max(true_spectrogram.max(), pred_spectrogram.max())

    # Display true spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(true_spectrogram, x_axis='time', y_axis='mel', sr=22050, vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('True Spectrogram')

    # Display predicted spectrogram
    plt.subplot(1, 2, 2)
    librosa.display.specshow(pred_spectrogram, x_axis='time', y_axis='mel', sr=22050, vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Predicted Spectrogram, Loss: {loss:.4f}')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory

def normalize_tensor(tensor):
    """Normalize a tensor to have mean 0 and std 1."""
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    return (tensor - mean) / (std + 1e-8)  # Add a small value to avoid division by zero

# Training Function
def train(model, train_loader, criterion, optimizer, num_epochs=10, base_dir='runs'):
    # Create a unique directory for this run
    os.makedirs(run_dir, exist_ok=True)  # Create the directory if it doesn't exist

    model = model.to(device)  # Move model to the appropriate device
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            #listen_to_waveform(labels[0].numpy(), fs=44100)

            # Normalize the inputs
            inputs = normalize_tensor(inputs)  # Normalize inputs
            
            # Normalize the labels
            labels = normalize_tensor(labels)  # Normalize labels
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm(2)  # L2 norm of the gradients
            #         print(f'Epoch [{epoch+1}/{num_epochs}], Layer: {name}, Gradient Norm: {grad_norm:.4f}')
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'training loss: {running_loss:.4f}\n')

    # Save final model
    save_model(model, model_path)

    # Save training losses to a file
    with open(os.path.join(run_dir, 'training_losses.txt'), 'w') as f:
        for loss in losses:
            f.write(f"{loss}\n")

    # Plot the loss curve and save the figure
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()  # Close the plot to free memory

def save_waveform(waveform, sample_rate, file_path):
    """Save the waveform as a WAV file."""
    # Normalize the waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform
    sf.write(file_path, waveform, sample_rate)
    
def test(model, test_loader, criterion, run_dir):
    model = model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            running_loss = 0.0
            # Normalize the inputs
            linear_spectrogram = librosa.feature.inverse.mel_to_stft(labels[i].detach().cpu().numpy(), sr=44100, n_fft=2048)
            reconstructed_waveform = librosa.istft(linear_spectrogram, hop_length=256)
            #listen_to_waveform(reconstructed_waveform, fs=44100)
          
            inputs = normalize_tensor(inputs)  # Normalize inputs
            
            # Normalize the labels
            labels = normalize_tensor(labels)  # Normalize labels
            # Calculate accuracy (if applicable)
            outputs = model(inputs)
            for j in range(len(inputs)):
                if j%10 == 0:
                    loss = criterion(outputs[j], labels[j])
                    # Accumulate loss
                    running_loss += loss.item() * inputs.size(0)
                    
                    true_spec = labels[j].detach().cpu().numpy()  # Example true spectrogram
                    linear_spectrogram = librosa.feature.inverse.mel_to_stft(true_spec, sr=44100, n_fft=2048)
                    reconstructed_waveform = librosa.istft(linear_spectrogram, hop_length=256)
                    waveform_file_path = os.path.join(run_dir, f'reconstructed_waveform_true_{j}.wav')
                    #listen_to_waveform(reconstructed_waveform, fs=44100)
                    save_waveform(reconstructed_waveform, 44100, waveform_file_path)
                
                    pred_spec = outputs[i].detach().cpu().numpy()  # Example predicted spectrogram
                    linear_spectrogram = librosa.feature.inverse.mel_to_stft(pred_spec, sr=44100, n_fft=2048)
                    reconstructed_waveform = librosa.istft(linear_spectrogram, hop_length=256)
                    waveform_file_path = os.path.join(run_dir, f'reconstructed_waveform_pred_{j}.wav')
                    save_waveform(reconstructed_waveform, 44100, waveform_file_path)
                    #listen_to_waveform(reconstructed_waveform, fs=44100)
                    
                    save_comparison_spectrograms(
                        true_spec,
                        pred_spec,
                        loss,
                        os.path.join(run_dir, f'comparison_spectrogram_{j + 1}.png')
                    )
            
        # Log the test loss
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'Test Loss: {running_loss/len(test_loader.dataset):.4f}\n')


def reconstruct_waveform(spectrogram, fs=44100):
    # Convert the spectrogram back to the time domain
    _, reconstructed_waveform = istft(spectrogram, fs=fs)
    return reconstructed_waveform

# def listen_to_waveform(waveform, fs=44100):
#     # Play the waveform using sounddevice
#     sd.play(waveform, samplerate=fs)
#     sd.wait()  # Wait until the sound has finished playing

# Example usage
if __name__ == "__main__":

    train_model = True
    #parameters to sweep
    in_channels = [14, 16, 22]
    patch_sizes = [50, 100, 200]
    num_heads = [4, 8, 16]
    num_layers = [2, 4, 8]
    ff_dims = [256, 512, 1024]
    patch_sizes = [50, 100, 200]
    num_epochs = [100,200,500,800,1000]
    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    criterion = [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()]

    for in_channel, patch_size, num_head, num_layer, ff_dim, num_epoch, learning_rate, criterion in itertools.product(in_channels, patch_sizes, num_heads, num_layers, ff_dims, num_epochs, learning_rates, criterion):


        # Initialize the model
        model = TransformerModel(in_channels=in_channel, embed_dim=256, num_heads=num_head, ff_dim=ff_dim, num_layers=num_layer, patch_size=patch_size)
        base_dir='swip_runs'
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f'run_{run_id}_num_channels{in_channel}_patch_{patch_size}_num_heads_{num_head}_num_layer_{num_layer}_ff_dim_{ff_dim}_num_epoch_{num_epoch}_learning_rate_{learning_rate}_criterion_{criterion}')
        # Define the log file path
        log_file = os.path.join(run_dir, 'loss_log.txt')
        model_path = os.path.join(run_dir, 'model.pth')
        
        # Ensure the run directory existstest_labels_tensor
        os.makedirs(run_dir, exist_ok=True)

        # Load data
        d_model = 256
        batch_size = 8
        sequence_length = 4000

        # Load data
        all_data = np.load('input_and_labels.npy',allow_pickle=True)

        # Get the total number of data points
        data_size = len(all_data)

        # Generate random indices
        random_indices = torch.randperm(data_size)

        # Calculate the number of samples to take (90%)
        train_size = int(data_size * 0.95)

        # Select random indices for training data
        train_indices = random_indices[:train_size]

        # Select random indices for testing data
        test_indices = random_indices[train_size:]

        # Create tensors for inputs and labels using the random indices
        train_inputs = [torch.tensor(np.array(all_data[i][0][:in_channel]), dtype=torch.float32) for i in train_indices]
        train_labels = [torch.tensor(np.array(all_data[i][1].astype(np.float32) / 32768.0), dtype=torch.float32) for i in train_indices]
        # Stack the tensors into a single tensor
        train_inputs_tensor = torch.stack(train_inputs)
        train_labels_tensor = torch.stack(train_labels)
        #for i in range(len(train_inputs_tensor)):
            #train_independent_components , train_ica = run_ica(np.transpose(train_inputs_tensor[i]),in_channel)
            #train_inputs_tensor[i] = torch.tensor(np.transpose(train_independent_components))
        #plot_signals(np.array(train_inputs_tensor[i]), np.transpose(train_independent_components))

        # Create tensors for test inputs and labels
        test_inputs = [torch.tensor(np.array(all_data[i][0][:in_channel]), dtype=torch.float32) for i in test_indices]
        test_labels = [torch.tensor(np.array(all_data[i][1].astype(np.float32) / 32768.0), dtype=torch.float32) for i in test_indices]

        # Stack the tensors into a single tensor
        test_inputs_tensor = torch.stack(test_inputs)
        test_labels_tensor = torch.stack(test_labels)
        #for i in range(len(test_inputs_tensor)):
            #test_independent_components , test_ica = run_ica(np.transpose(test_inputs_tensor[i]),in_channel)
            #test_inputs_tensor[i] = torch.tensor(np.transpose(test_independent_components))


        # Process the labels to be spectograms
        train_spectrograms = process_labels(train_labels_tensor)
        train_spectrograms_tensor = torch.tensor(train_spectrograms, dtype=torch.float32)
        test_spectrograms = process_labels(test_labels_tensor)
        test_spectrograms_tensor = torch.tensor(test_spectrograms, dtype=torch.float32)

        # Create DataLoader
        train_dataset = TensorDataset(train_inputs_tensor, train_spectrograms_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_inputs_tensor, test_spectrograms_tensor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

        # Loss and optimizer
        criterion = criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        if train_model:
            # Train the model
            train(model, train_loader, criterion, optimizer, num_epochs=num_epoch, base_dir=base_dir)
        else:
            # Load the model
            model.load_state_dict(torch.load(model_path))

        # Test the model
        test(model, test_loader, criterion, run_dir)
