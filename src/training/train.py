
from src.datasets.PhysioNetMI_preprocess import preprocess_spd



filter_bank = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]



X, y, class_names = preprocess_spd(
        filter_bank=filter_bank,
        estimator="lwf",
        sfreq=160,
        eps=1e-6,
        segment_duration=1,
        stride_duration=0.5,
    )



print(f"X.shape: {X.shape}")
print(X[:1])
print(y.shape)
print(class_names)
