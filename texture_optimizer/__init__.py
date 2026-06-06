"""
texture_optimizer
=================
Per-camera PPISP + texture joint optimiser for COLMAP scenes.

Quick start:
    from texture_optimizer import train_scene
    train_scene("path/to/colmap_scene", output_dir="outputs", num_iterations=5000)
"""

from .ppisp    import PPISPParams
from .renderer import TextureMap, Rasterizer, NvdiffrastRasterizer, SoftwareRasterizer
from .losses   import TotalLoss, PhotometricLoss, PPISPRegLoss
from .dataset  import ColmapScene, CameraView, MeshData, load_obj
from .trainer  import TexturePPISPTrainer, TrainConfig, train_scene

__all__ = [
    "PPISPParams",
    "TextureMap", "Rasterizer", "NvdiffrastRasterizer", "SoftwareRasterizer",
    "TotalLoss", "PhotometricLoss", "PPISPRegLoss",
    "ColmapScene", "CameraView", "MeshData", "load_obj",
    "TexturePPISPTrainer", "TrainConfig", "train_scene",
]
