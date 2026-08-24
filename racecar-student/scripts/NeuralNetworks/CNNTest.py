import numpy as np
import torch

from train_lidar_denoiser import (
    LidarDenoiser,
    MAX_DISTANCE_CM,
    add_snow_noise
)


MODEL_PATH = "lidar_denoiser.pt"


model = LidarDenoiser()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


#Load one clean scan
clean = np.load("clean_scans.npy")[0]

#Artificially corrupt it
noisy = add_snow_noise(clean)


x = noisy / MAX_DISTANCE_CM

x = torch.tensor(
    x,
    dtype=torch.float32
)

x = x.unsqueeze(0)
x = x.unsqueeze(0)


with torch.no_grad():

    prediction = model(x)


prediction = prediction.squeeze().numpy()

prediction *= MAX_DISTANCE_CM

prediction = np.clip(
    prediction,
    0,
    MAX_DISTANCE_CM
)


print("Noisy:")
print(noisy)

print("\nClean:")
print(clean)

print("\nPredicted:")
print(prediction)