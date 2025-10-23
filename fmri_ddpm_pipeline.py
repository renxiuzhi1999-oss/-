"""Comprehensive fMRI DDPM training and generation script.

This module provides dataset handling, UNet backbone, diffusion processes,
training utilities, and a CLI for training or generating conditioned samples.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------- Dataset utilities ---------------------------- #

@dataclass
class SubjectMetadata:
    subject_id: str
    age: float
    gender: str
    label: str

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "SubjectMetadata":
        return cls(
            subject_id=str(record.get("subject_id")),
            age=float(record.get("age")),
            gender=str(record.get("gender")),
            label=str(record.get("label")),
        )


class FMRIExample(Dataset):
    """Dataset wrapper for fMRI volumes with demographic conditioning."""

    def __init__(
        self,
        data_dir: Path,
        metadata_file: Path,
        volume_shape: Tuple[int, int, int],
        gender_map: Optional[Dict[str, int]] = None,
        label_map: Optional[Dict[str, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)
        self.volume_shape = volume_shape
        self.dtype = dtype
        self.records = self._load_metadata()
        self.gender_map = gender_map or self._build_map("gender")
        self.label_map = label_map or self._build_map("label")

    def _load_metadata(self) -> List[Dict[str, Any]]:
        with open(self.metadata_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Metadata JSON must be a list of records")
        return payload

    def _build_map(self, key: str) -> Dict[str, int]:
        values = sorted({str(item.get(key)) for item in self.records})
        return {value: idx for idx, value in enumerate(values)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        metadata = SubjectMetadata.from_record(record)

        path = self.data_dir / f"{metadata.subject_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Volume file not found for subject {metadata.subject_id}")
        volume = np.load(path).astype(np.float32)
        volume = volume.reshape(self.volume_shape)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
        volume = torch.from_numpy(volume).unsqueeze(0).to(self.dtype)

        gender_idx = self.gender_map[metadata.gender]
        label_idx = self.label_map[metadata.label]
        cond = torch.tensor([metadata.age, gender_idx, label_idx], dtype=self.dtype)

        return {
            "volume": volume,
            "conditioning": cond,
            "gender_index": gender_idx,
            "metadata": metadata,
        }


# ----------------------------- Diffusion utils ----------------------------- #

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None, None]
        h = h + self.cond_mlp(cond_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class UNet3D(nn.Module):
    def __init__(self, channels: int, base_channels: int, channel_multipliers: Iterable[int], time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.init_conv = nn.Conv3d(1, base_channels, kernel_size=3, padding=1)

        downs = []
        in_channels = base_channels
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        for mult in channel_multipliers:
            out_channels = base_channels * mult
            downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_channels, out_channels, time_dim, cond_dim),
                        ResidualBlock(out_channels, out_channels, time_dim, cond_dim),
                        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    ]
                )
            )
            in_channels = out_channels
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList(
            [
                ResidualBlock(in_channels, in_channels, time_dim, cond_dim),
                ResidualBlock(in_channels, in_channels, time_dim, cond_dim),
            ]
        )

        ups = []
        for mult in reversed(list(channel_multipliers)):
            out_channels = base_channels * mult
            ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_channels * 2, out_channels, time_dim, cond_dim),
                        ResidualBlock(out_channels, out_channels, time_dim, cond_dim),
                        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    ]
                )
            )
            in_channels = out_channels
        self.ups = nn.ModuleList(ups)

        self.final_block = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, channels, kernel_size=3, padding=1),
        )
        self.time_embeddings = SinusoidalPositionEmbeddings(time_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embeddings(time)
        h = self.init_conv(x)
        residuals = []
        for block1, block2, downsample in self.downs:
            h = block1(h, t_emb, cond)
            h = block2(h, t_emb, cond)
            residuals.append(h)
            h = downsample(h)

        for block in self.mid:
            h = block(h, t_emb, cond)

        for block1, block2, upsample in self.ups:
            res = residuals.pop()
            h = torch.cat((h, res), dim=1)
            h = block1(h, t_emb, cond)
            h = block2(h, t_emb, cond)
            h = upsample(h)

        return self.final_block(h)


class LinearNoiseScheduler:
    def __init__(self, num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps].sqrt().to(original_samples.device)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]).sqrt().to(original_samples.device)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., None]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., None]
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


# ----------------------------- Training helpers --------------------------- #

class DDPMTrainer:
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: LinearNoiseScheduler,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _condition_embedding(self, conditioning: torch.Tensor) -> torch.Tensor:
        cond_mean = conditioning.mean(dim=0, keepdim=True)
        cond_std = conditioning.std(dim=0, keepdim=True) + 1e-6
        return (conditioning - cond_mean) / cond_std

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            volumes = batch["volume"].to(self.device)
            conditioning = batch["conditioning"].to(self.device)
            cond_emb = self._condition_embedding(conditioning)

            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (volumes.size(0),), device=self.device).long()
            noise = torch.randn_like(volumes)
            noisy_volumes = self.noise_scheduler.add_noise(volumes, noise, timesteps)

            predicted_noise = self.model(noisy_volumes, timesteps, cond_emb)
            loss = self.criterion(predicted_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * volumes.size(0)
        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        sample_shape: Tuple[int, int, int],
        base_conditioning: torch.Tensor,
        gender_index: int,
    ) -> torch.Tensor:
        self.model.eval()
        cond_emb = self._condition_embedding(base_conditioning)
        opposite_gender = 1 - gender_index  # assumes binary genders encoded as 0/1
        cond_emb[:, 1] = (opposite_gender - cond_emb[:, 1].mean()) / (cond_emb[:, 1].std() + 1e-6)

        device = self.device
        samples = torch.randn((num_samples, 1, *sample_shape), device=device)
        for timestep in reversed(range(self.noise_scheduler.num_train_timesteps)):
            t = torch.full((num_samples,), timestep, device=device, dtype=torch.long)
            noise_pred = self.model(samples, t, cond_emb)
            beta = self.noise_scheduler.betas[timestep].to(device)
            alpha = self.noise_scheduler.alphas[timestep].to(device)
            alpha_cum = self.noise_scheduler.alphas_cumprod[timestep].to(device)
            samples = (1 / alpha.sqrt()) * (samples - (beta / (1 - alpha_cum).sqrt()) * noise_pred)
            if timestep > 0:
                samples = samples + beta.sqrt() * torch.randn_like(samples)
        return samples.clamp(-1, 1)


# ----------------------------- CLI Entrypoints ---------------------------- #

@dataclass
class TrainConfig:
    data_dir: Path
    metadata_path: Path
    volume_shape: Tuple[int, int, int]
    batch_size: int
    epochs: int
    learning_rate: float
    timesteps: int
    device: str


def create_dataloader(config: TrainConfig) -> Tuple[DataLoader, FMRIExample]:
    dataset = FMRIExample(
        data_dir=config.data_dir,
        metadata_file=config.metadata_path,
        volume_shape=config.volume_shape,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    return dataloader, dataset


def train(config: TrainConfig) -> None:
    dataloader, dataset = create_dataloader(config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = UNet3D(
        channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        time_dim=128,
        cond_dim=3,
    ).to(device)
    noise_scheduler = LinearNoiseScheduler(num_train_timesteps=config.timesteps)
    trainer = DDPMTrainer(model, noise_scheduler, learning_rate=config.learning_rate, device=device)

    for epoch in range(1, config.epochs + 1):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch}: loss={loss:.4f}")
        torch.save({
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "gender_map": dataset.gender_map,
            "label_map": dataset.label_map,
        }, config.data_dir / f"ddpm_checkpoint_epoch_{epoch}.pt")


def generate(
    checkpoint_path: Path,
    metadata_path: Path,
    subject_id: str,
    output_path: Path,
    volume_shape: Tuple[int, int, int],
    timesteps: int,
    device: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet3D(
        channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        time_dim=128,
        cond_dim=3,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    noise_scheduler = LinearNoiseScheduler(num_train_timesteps=timesteps)
    trainer = DDPMTrainer(model, noise_scheduler, learning_rate=1e-4, device=torch.device(device))

    with open(metadata_path, "r", encoding="utf-8") as handle:
        records = {str(item["subject_id"]): item for item in json.load(handle)}
    record = records[subject_id]
    metadata = SubjectMetadata.from_record(record)
    gender_idx = checkpoint["gender_map"][metadata.gender]

    cond = torch.tensor([[metadata.age, gender_idx, checkpoint["label_map"][metadata.label]]], dtype=torch.float32)
    generated = trainer.generate(num_samples=1, sample_shape=volume_shape, base_conditioning=cond, gender_index=gender_idx)

    np.save(output_path, generated.squeeze(0).cpu().numpy())
    print(f"Saved generated volume to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run the fMRI DDPM generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("data_dir", type=Path)
    train_parser.add_argument("metadata_path", type=Path)
    train_parser.add_argument("--volume-shape", type=int, nargs=3, default=(64, 64, 64))
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--timesteps", type=int, default=1000)
    train_parser.add_argument("--device", type=str, default="cuda")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("checkpoint", type=Path)
    gen_parser.add_argument("metadata_path", type=Path)
    gen_parser.add_argument("subject_id", type=str)
    gen_parser.add_argument("output_path", type=Path)
    gen_parser.add_argument("--volume-shape", type=int, nargs=3, default=(64, 64, 64))
    gen_parser.add_argument("--timesteps", type=int, default=1000)
    gen_parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        config = TrainConfig(
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            volume_shape=tuple(args.volume_shape),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            timesteps=args.timesteps,
            device=args.device,
        )
        train(config)
    elif args.command == "generate":
        generate(
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata_path,
            subject_id=args.subject_id,
            output_path=args.output_path,
            volume_shape=tuple(args.volume_shape),
            timesteps=args.timesteps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
