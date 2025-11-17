"""
Orchestrates training of all models.
"""

import subprocess

if __name__ == "__main__":
    lat_dim = 32

    cmd_train_tsc_vae = [
        "python", "./Model/TSC_vae.py",
        "--latent_dim", str(lat_dim),
        "--batch_size", str(16),
        "--epochs", str(500),
        "--learning_rate", str(1e-4)
    ]

    cmd_train_vae = [
        "python", "./FT_VAE.py",
        "--latent_dim", str(lat_dim),
        "--batch_size", str(16),
        "--num_heads", str(4),
        "--epochs", str(100),
        "--learning_rate", str(1e-4)
    ]


    cmd_test = ["python", "./FT_VAE_testing.py", "--latent_dim", str(lat_dim)]
   
    print("Start Training TSC-VAE...")
    subprocess.run(cmd_train_tsc_vae)
    print("Finished! \n")

    print("Start Training PepEmbedNet VAE...")
    subprocess.run(cmd_train_vae)
    print("Finished! \n")

    print("Start Correlation Tests...")
    subprocess.run(cmd_test)
    print("Finished!")
