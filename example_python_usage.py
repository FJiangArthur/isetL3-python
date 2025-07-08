"""Small example demonstrating the Python port."""

import numpy as np
from isetL3.classify import SimpleClassifier
from isetL3.data import L3DataCamera
from isetL3.train import L3TrainOLS
from isetL3.render import L3Render


# Create synthetic raw/target data
raw = np.random.rand(8, 8)
rgb = np.stack([raw, raw, raw], axis=-1)

# simple CFA map
cfa = np.array([[1, 2], [3, 4]])

# Build data class
l3d = L3DataCamera([raw], [rgb], cfa)

# Classification + training
classifier = SimpleClassifier(patch_size=5)
trainer = L3TrainOLS(l3c=classifier)
trainer.train(l3d)

# Render
renderer = L3Render()
rendered = renderer.render(raw, l3d.p_type, trainer)
print("Rendered shape:", rendered.shape)
