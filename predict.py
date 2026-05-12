import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/src/hf-cache"

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

WEIGHTS_DIR = "/src/hf-cache"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(WEIGHTS_DIR))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)

        print(f"[setup] importing terratorch... (t={time.time()-t0:.1f}s)", flush=True)
        from terratorch.registry import FULL_MODEL_REGISTRY
        self.FULL_MODEL_REGISTRY = FULL_MODEL_REGISTRY

        print(f"[setup] building terramind_v1_tokenizer_ndvi... (t={time.time()-t0:.1f}s)", flush=True)
        self.model = FULL_MODEL_REGISTRY.build(
            "terramind_v1_tokenizer_ndvi",
            pretrained=True,
        )
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.float32)
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def _load_ndvi(self, image_path: Path) -> torch.Tensor:
        """
        Carrega imagem NDVI como tensor [B=1, C=1, H, W] em float32 [-1, 1].
        Aceita:
        - PNG/JPG grayscale onde pixel value = NDVI normalizado
        - PNG/JPG colorido (toma média dos canais como aproximação)
        - GeoTIFF (.tif/.tiff) single-band
        """
        path_str = str(image_path)
        if path_str.lower().endswith((".tif", ".tiff")):
            try:
                import rasterio
                with rasterio.open(path_str) as src:
                    arr = src.read(1).astype(np.float32)
            except Exception as e:
                print(f"[predict] rasterio fail, fallback PIL: {e}", flush=True)
                arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        else:
            pil = Image.open(image_path).convert("L")
            arr = np.asarray(pil, dtype=np.float32)

        # Normaliza: se valores estão em [0, 255], mapeia pra [-1, 1] (NDVI standard)
        if arr.max() > 1.0:
            arr = (arr / 127.5) - 1.0  # [0, 255] -> [-1, 1]
        else:
            arr = (arr * 2.0) - 1.0  # [0, 1] -> [-1, 1]

        # [H, W] -> [1, 1, H, W]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict(
        self,
        image: Path = Input(
            description="Imagem NDVI (grayscale PNG/JPG ou GeoTIFF). Valores [0-255] mapeados pra NDVI [-1, 1].",
        ),
        mode: str = Input(
            description="Modo de operação: 'tokenize' (encode em tokens), 'reconstruct' (encode→decode), 'tokens_only' (só índices).",
            default="reconstruct",
            choices=["tokenize", "reconstruct", "tokens_only"],
        ),
        timesteps: int = Input(
            description="Diffusion steps no decoder (mais steps = melhor reconstrução, mais lento).",
            default=10,
            ge=1,
            le=50,
        ),
        image_size: int = Input(
            description="Resolução de entrada (deve ser múltiplo de patch size).",
            default=224,
            ge=64,
            le=1024,
        ),
    ) -> str:
        device = next(self.model.parameters()).device

        # Carrega NDVI e redimensiona
        ndvi_tensor = self._load_ndvi(image)
        # Resize via interpolate pra image_size
        ndvi_tensor = torch.nn.functional.interpolate(
            ndvi_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        ndvi_tensor = ndvi_tensor.to(device, dtype=torch.float32)

        with torch.no_grad():
            if mode == "tokens_only":
                _, _, tokens = self.model.encode(ndvi_tensor)
                result = {
                    "mode": "tokens_only",
                    "image_size": image_size,
                    "n_tokens": int(tokens.numel()),
                    "token_shape": list(tokens.shape),
                    "tokens": tokens.flatten().cpu().tolist(),
                }
            elif mode == "tokenize":
                _, _, tokens = self.model.encode(ndvi_tensor)
                result = {
                    "mode": "tokenize",
                    "image_size": image_size,
                    "n_tokens": int(tokens.numel()),
                    "token_shape": list(tokens.shape),
                    "tokens_preview": tokens.flatten()[:64].cpu().tolist(),
                    "tokens_min": int(tokens.min()),
                    "tokens_max": int(tokens.max()),
                }
            else:  # reconstruct
                _, _, tokens = self.model.encode(ndvi_tensor)
                reconstruction = self.model.decode_tokens(tokens, verbose=False, timesteps=timesteps)
                recon_np = reconstruction.squeeze().cpu().numpy()
                orig_np = ndvi_tensor.squeeze().cpu().numpy()

                # MSE de reconstrução
                mse = float(np.mean((recon_np - orig_np) ** 2))
                psnr = float(-10 * np.log10(max(mse, 1e-12)))

                result = {
                    "mode": "reconstruct",
                    "image_size": image_size,
                    "timesteps": timesteps,
                    "n_tokens": int(tokens.numel()),
                    "mse": mse,
                    "psnr_db": psnr,
                    "reconstruction_stats": {
                        "min": float(recon_np.min()),
                        "max": float(recon_np.max()),
                        "mean": float(recon_np.mean()),
                        "std": float(recon_np.std()),
                    },
                    "original_stats": {
                        "min": float(orig_np.min()),
                        "max": float(orig_np.max()),
                        "mean": float(orig_np.mean()),
                        "std": float(orig_np.std()),
                    },
                }

        return json.dumps(result, ensure_ascii=False)
