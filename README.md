# terramind-ndvi-tokenizer

Deploy do **[ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI)** no Replicate. Tokenizer FSQ-VAE para imagens NDVI (Normalized Difference Vegetation Index) — componente do TerraMind.

## Modelo
- **Tipo**: FSQ-VAE Tokenizer (encode/decode NDVI)
- **Codebook**: 15.360 tokens, 5 dimensões FSQ
- **Decoder**: usa diffusion steps configuráveis
- **Pré-treinamento**: 20 épocas em 9M imagens NDVI (TerraMesh dataset)
- **Licença**: Apache 2.0

## API

### Modos

- **`tokenize`**: encode NDVI → preview de tokens + estatísticas
- **`tokens_only`**: encode NDVI → todos os tokens (lista completa)
- **`reconstruct`**: encode → decode com diffusion (avalia qualidade via PSNR/MSE)

### Inputs

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `image` | Path | obrigatório | NDVI grayscale (PNG/JPG/TIFF). Valores 0-255 mapeados pra [-1, 1] |
| `mode` | string | "reconstruct" | tokenize / tokens_only / reconstruct |
| `timesteps` | int | 10 | Diffusion steps no decoder (1-50) |
| `image_size` | int | 224 | Resolução de entrada (64-1024) |

### Output (reconstruct)

```json
{
  "mode": "reconstruct",
  "image_size": 224,
  "timesteps": 10,
  "n_tokens": 196,
  "mse": 0.0024,
  "psnr_db": 26.21,
  "reconstruction_stats": {"min": -0.92, "max": 0.88, "mean": 0.12, "std": 0.41},
  "original_stats": {"min": -1.0, "max": 1.0, "mean": 0.11, "std": 0.45}
}
```

### Output (tokenize)

```json
{
  "mode": "tokenize",
  "n_tokens": 196,
  "token_shape": [1, 14, 14],
  "tokens_preview": [42, 1023, ...],
  "tokens_min": 0,
  "tokens_max": 15359
}
```

## Hardware
- **gpu-t4** (16 GB) — modelo pequeno
- Encode: ~1-2 s
- Reconstruct (10 timesteps): ~3-5 s
