import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#TODO: change to size 1080 scan for rplidar, use sim to collect lidar scans for preliminary training.
#TODO: install dependencies locally and on racecar environment.
#instructions: pip install torch numpy onnx onnxruntime
#uploading to racecar: in the env, pip install onnxruntime
#also, collect actual snow data. artificial will not suffice.


DATA_PATH = "clean_scans.npy"

MAX_DISTANCE_CM = 400.0

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

MODEL_PATH = "lidar_denoiser.pt"
ONNX_PATH = "lidar_denoiser.onnx"



def add_snow_noise(scan):
    noisy = scan.copy()

    probability = 0.015

    mask = np.random.random(720) < probability

    random_returns = np.random.uniform(
        5,
        MAX_DISTANCE_CM,
        size=720
    )

    noisy[mask] = random_returns[mask]
    num_clusters = np.random.randint(0, 5)

    for _ in range(num_clusters):

        start = np.random.randint(0, 720)

        length = np.random.randint(1, 6)

        end = min(start + length, 720)

        noisy[start:end] = np.random.uniform(
            5,
            MAX_DISTANCE_CM,
            size=end - start
        )
    invalid_probability = 0.005

    mask = np.random.random(720) < invalid_probability

    noisy[mask] = MAX_DISTANCE_CM

    noisy = np.clip(
        noisy,
        0,
        MAX_DISTANCE_CM
    )

    return noisy

class LidarDataset(Dataset):

    def __init__(self, clean_scans):

        self.clean = np.asarray(
            clean_scans,
            dtype=np.float32
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):

        clean = self.clean[idx]

        noisy = add_snow_noise(clean)

        clean = clean / MAX_DISTANCE_CM
        noisy = noisy / MAX_DISTANCE_CM

        noisy = torch.tensor(
            noisy,
            dtype=torch.float32
        ).unsqueeze(0)

        clean = torch.tensor(
            clean,
            dtype=torch.float32
        ).unsqueeze(0)

        return noisy, clean


class LidarDenoiser(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.Conv1d(
                32,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),      


            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.MaxPool1d(2)       
        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),                      

            nn.ReLU(),

            nn.Conv1d(
                32,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.ConvTranspose1d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),                   

            nn.ReLU(),

            nn.Conv1d(
                16,
                1,
                kernel_size=7,
                padding=3
            )
        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x

def main():

    print("Loading data...")

    clean_scans = np.load(DATA_PATH)

    print("Dataset shape:", clean_scans.shape)

    if clean_scans.ndim != 2:
        raise ValueError(
            "Expected data shape (N, 720)"
        )

    if clean_scans.shape[1] != 720:
        raise ValueError(
            "Expected exactly 720 LiDAR samples"
        )


    dataset = LidarDataset(clean_scans)

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using:", device)


    model = LidarDenoiser().to(device)


    # Huber loss
    criterion = nn.HuberLoss(
        delta=0.05
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for noisy, clean in train_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            prediction = model(noisy)

            loss = criterion(
                prediction,
                clean
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()


        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for noisy, clean in val_loader:

                noisy = noisy.to(device)
                clean = clean.to(device)

                prediction = model(noisy)

                loss = criterion(
                    prediction,
                    clean
                )

                val_loss += loss.item()


        val_loss /= len(val_loader)


        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        "Saved:",
        MODEL_PATH
    )

    model.eval()

    dummy_input = torch.randn(
        1,
        1,
        720
    ).to(device)


    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,

        input_names=["lidar"],
        output_names=["clean_lidar"],

        dynamic_axes={
            "lidar": {
                0: "batch"
            },

            "clean_lidar": {
                0: "batch"
            }
        },

        opset_version=17
    )


    print(
        "Saved:",
        ONNX_PATH
    )


if __name__ == "__main__":
    main()