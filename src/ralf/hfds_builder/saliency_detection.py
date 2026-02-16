import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image
from tqdm import tqdm

from ralf.hfds_builder.models.saliency.basnet import BASNet, RescaleT, ToTensorLab
from ralf.hfds_builder.models.saliency.isnet import ISNetDIS
from ralf.train.global_variables import CACHE_DIR

logger = logging.getLogger(__name__)


WEIGHT_ROOT = Path(CACHE_DIR) / "hfds_builder" / "saliency_detection"
assert WEIGHT_ROOT.exists(), f"{str(WEIGHT_ROOT.resolve())} does not exist."


def main(
    input_dir: str | None = None,
    output_dir: str | None = None,
    algorithm: str = "isnet",
    input_ext: str | None = None,
    weight_dir: str | None = None,
) -> None:
    global WEIGHT_ROOT
    if input_dir is None or output_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_dir",
            required=True,
            type=str,
            help="Input file path.",
        )
        parser.add_argument(
            "--output_dir",
            required=True,
            type=str,
            help="Output prefix.",
        )
        parser.add_argument(
            "--input_ext",
            type=str,
            help="Limit the type of input file extension.",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            choices=["isnet", "basnet"],
            default="isnet",
        )
        parser.add_argument(
            "--weight_dir",
            type=str,
            default=str(WEIGHT_ROOT),
        )
        args = parser.parse_args()
        input_dir = args.input_dir
        output_dir = args.output_dir
        algorithm = args.algorithm
        input_ext = args.input_ext
        weight_dir = args.weight_dir

    if weight_dir:
        WEIGHT_ROOT = Path(weight_dir)

    logger.info(f"input_dir={input_dir} output_dir={output_dir} algorithm={algorithm}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == "isnet":
        tester = ISNetSaliencyTester()
    elif algorithm == "basnet":
        tester = BASNetSaliencyTester()
    else:
        raise NotImplementedError

    pattern = f"*.{input_ext}" if input_ext else "*"
    for input_path in tqdm(list(Path(input_dir).glob(pattern))):
        if not input_path.is_file():
            continue

        output_path = output_dir / input_path.name
        if output_path.exists():
            # if there already exists the output file, skip
            continue

        image = Image.open(input_path).convert("RGB")
        width, height = image.size

        pred = tester(image)
        pred = torch.squeeze(F.interpolate(pred, (height, width), mode="bilinear"), 0)
        pred = _norm_pred(pred)

        logger.info(f"{input_path=} {output_path=}")
        with output_path.open("wb") as f:
            save_image(pred, f)


class _SaliencyTester:  # type: ignore
    def __init__(self) -> None:
        self._model: nn.Module = nn.Identity()  # to be overwritten
        self._ckpt_path: str = ""  # to be overwritten

    def setup_model(self, model: nn.Module) -> None:
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"))
        model.eval()
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        self._model = model

    def __call__(self, image: Image) -> None:
        raise NotImplementedError


class ISNetSaliencyTester(_SaliencyTester):  # type: ignore
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self._transform = ToTensor()
        self._input_size = (1024, 1024)
        self._ckpt_path = str(WEIGHT_ROOT / "isnet-general-use.pth")
        self.setup_model(ISNetDIS())  # type: ignore

    @torch.no_grad()
    def __call__(self, image: Image) -> Tensor:
        # preprocess
        width, height = image.size
        img = self._transform(image).unsqueeze(0)
        img = F.interpolate(img, self._input_size, mode="bilinear")
        img = normalize(img, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

        # prediction
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda"))
        pred = self._model(img)[0][0].cpu().detach()
        assert list(pred.size()) == [1, 1, 1024, 1024]

        return pred
        # self.postprocess_and_save_image(pred=pred, path=path, width=width, height=height)


class BASNetSaliencyTester(_SaliencyTester):  # type: ignore
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # note: this transforms takes and returns numpy in np.uint8 with size (H, W, C)
        self._transform = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])  # type: ignore
        self._ckpt_path = str(WEIGHT_ROOT / "gdi-basnet.pth")
        self.setup_model(BASNet(3, 1))  # type: ignore

    @torch.no_grad()
    def __call__(self, img_pil: Image) -> Tensor:
        # preprocess
        img_npy = np.array(img_pil, dtype=np.uint8)
        width, height = img_pil.size
        assert img_npy.shape[-1] == 3
        label_npy = np.zeros((height, width), dtype=np.uint8)
        img = self._transform({"image": img_npy, "label": label_npy})["image"]
        img = img.float().unsqueeze(0)

        # prediction
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda"))
        pred = self._model(img)[0].cpu().detach()[:, 0].unsqueeze(0)
        assert list(pred.size()) == [1, 1, 256, 256]

        return pred
        # self.postprocess_and_save_image(pred=pred, path=path, width=width, height=height)


def _norm_pred(d: Tensor) -> Tensor:
    ma = torch.max(d)
    mi = torch.min(d)
    # division while avoiding zero division
    dn = (d - mi) / ((ma - mi) + torch.finfo(torch.float32).eps)
    return dn


if __name__ == "__main__":
    main()
