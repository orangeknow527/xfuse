"""Train the XFuse ST model on patch-level HE images and gene counts.

The script expects a directory of PNG patches (e.g., ``H1_patches/3x25.png``)
and a ``count.csv`` whose first column contains spot identifiers such as
``GE283648_3x25``. The substring after the underscore is matched to patch
filenames. Images are resized to 256x256, normalized to ``[-1, 1]``, and paired
with their gene expression vectors. After training, pooled bottleneck vectors
are saved for every patch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyro
import torch
from PIL import Image
from torchvision import transforms

from xfuse.data.utility.misc import make_dataloader
from xfuse.model import XFuse
from xfuse.model.experiment.st import ST
from xfuse.optim import Adam
from xfuse.session import Session
from xfuse.train import train


def _load_counts(counts_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    counts = pd.read_csv(counts_csv)
    if counts.shape[1] < 2:
        raise ValueError("count.csv must contain spot IDs and at least one gene column")

    spot_ids = counts.iloc[:, 0].astype(str).str.split("_", n=1).str[-1]
    counts.index = spot_ids
    genes = counts.columns[1:].tolist()
    expression = counts.iloc[:, 1:]
    return expression, genes


class PatchGeneDataset(torch.utils.data.Dataset):
    """Simple dataset that matches patch images with gene expression vectors."""

    def __init__(self, patch_dir: Path, counts_csv: Path, target_size: int = 256, margin: int = 4):
        self.patch_dir = patch_dir
        self.target_size = target_size
        self.margin = margin

        expression, genes = _load_counts(counts_csv)
        self.genes = genes

        samples = []
        for img_path in sorted(self.patch_dir.glob("*.png")):
            name = img_path.stem
            if name not in expression.index:
                continue
            samples.append((img_path, name, torch.as_tensor(expression.loc[name].values, dtype=torch.float32)))

        if len(samples) == 0:
            raise RuntimeError("No matching patch images and gene expressions were found.")

        self.samples = samples
        self.transform = transforms.Compose(
            [transforms.Resize((self.target_size, self.target_size)), transforms.ToTensor(),]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def size(self, data_type=None, slide=None, covariate=None, condition=None) -> int:  # pylint: disable=unused-argument
        return len(self.samples)

    def _make_label(self) -> torch.Tensor:
        label = torch.zeros((self.target_size, self.target_size), dtype=torch.long)
        if self.margin >= self.target_size // 2:
            label[...] = 1
            return label
        label[self.margin : self.target_size - self.margin, self.margin : self.target_size - self.margin] = 1
        return label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, name, gene_vector = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        image = image * 2 - 1  # scale to [-1, 1]

        label = self._make_label()
        data = gene_vector.unsqueeze(0)

        return {
            "data_type": "ST",
            "image": image,
            "label": label,
            "data": data,
            "slide": "H1",
            "covariates": {},
            "name": name,
        }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XFuse ST model on H1 patch dataset")
    parser.add_argument("patch_dir", type=Path, help="Directory containing spot patch PNG images")
    parser.add_argument("counts_csv", type=Path, help="CSV file with spot-level gene counts")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size", help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, dest="learning_rate", help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, dest="num_workers", help="DataLoader worker count")
    parser.add_argument("--device", type=str, default=None, help="Torch device to use (e.g., cuda or cpu)")
    parser.add_argument(
        "--feature-out",
        type=Path,
        default=Path("bottleneck_features.pt"),
        dest="feature_out",
        help="Path to save pooled bottleneck vectors",
    )
    parser.add_argument("--network-depth", type=int, default=4, dest="network_depth", help="Encoder/decoder depth")
    parser.add_argument("--network-width", type=int, default=8, dest="network_width", help="Base channel width")
    return parser


def _extract_features(model: XFuse, dataset: PatchGeneDataset, genes: List[str], device: torch.device, out_path: Path):
    model.eval()
    experiment = model.get_experiment("ST")
    feature_map: Dict[str, torch.Tensor] = {}

    with torch.no_grad(), Session(model=model, genes=genes, messengers=[]):
        for sample in dataset:
            x = {
                "image": sample["image"].unsqueeze(0).to(device),
                "label": sample["label"].unsqueeze(0).to(device),
                "data": sample["data"].to(device),
                "slide": [sample["slide"]],
                "covariates": [sample["covariates"]],
            }
            zs = experiment.guide(x)
            bottleneck = zs[-1].detach().cpu()
            feature_map[sample["name"]] = bottleneck.mean(dim=(2, 3)).squeeze(0)

    torch.save(feature_map, out_path)


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = PatchGeneDataset(args.patch_dir, args.counts_csv)
    dataloader = make_dataloader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = XFuse(experiments=[ST(depth=args.network_depth, num_channels=args.network_width)]).to(device)
    optimizer = Adam({"lr": args.learning_rate, "amsgrad": True})

    pyro.clear_param_store()

    with Session(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        genes=dataset.genes,
        learning_rate=args.learning_rate,
        default_device=device,
    ):
        train(args.epochs)

    _extract_features(model, dataset, dataset.genes, device, args.feature_out)


if __name__ == "__main__":
    main()
