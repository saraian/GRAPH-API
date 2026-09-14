import os

import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder


# The exported l2 encoder consumes a 512x512 padded image, but the EfficientViT SAM
# checkpoint keeps the prompt encoder/mask post-processing coordinate frame at 1024x1024.
# These are deliberately separate sizes: the image is encoded at 512, while VLM boxes are
# transformed to 1024 by SamDecoder before the decoder is called and its mask is restored.
VITSAM_IMAGE_SIZE = 512
VITSAM_PROMPT_IMAGE_SIZE = 1024


class VitSam():

    def __init__(self, encoder_model, decoder_model):
        # VitSAM is an ONNX model. Select the device from ONNX Runtime's providers, not
        # from torch.cuda.is_available(): PyTorch and ONNX Runtime can have different CUDA
        # installations, and the latter is the runtime that executes these two models.
        import onnxruntime as ort

        require_cuda = os.environ.get("VITSAM_REQUIRE_CUDA", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        if require_cuda and not cuda_available:
            raise RuntimeError(
                "VitSAM requires CUDA, but ONNX Runtime does not expose "
                "CUDAExecutionProvider"
            )
        requested_device = "cuda" if cuda_available else "cpu"
        self.device = requested_device

        def make_sessions(device):
            return (
                SamDecoder(decoder_model, device=device, target_size=VITSAM_PROMPT_IMAGE_SIZE),
                SamEncoder(encoder_model, device=device),
            )

        # The encoder and decoder intentionally use different sizes for this exported model:
        # l2's encoder input is 512, while its prompt/mask frame is 1024 (see efficientvit's
        # image_size=(1024, 512) definition). Do not use VITSAM_IMAGE_SIZE for target_size.
        self.decoder, self.encoder = make_sessions(requested_device)

        # ONNX Runtime may list CUDAExecutionProvider but silently fall back to CPU when
        # one of its shared-library dependencies or NVIDIA device permissions is missing.
        # Check the providers selected by both actual sessions so the log never claims GPU
        # execution when VitSAM is really running on CPU.
        decoder_providers = set(self.decoder.session.get_providers())
        encoder_providers = set(self.encoder.session.get_providers())
        cuda_sessions_ok = (
            "CUDAExecutionProvider" in decoder_providers
            and "CUDAExecutionProvider" in encoder_providers
        )
        if requested_device == "cuda" and not cuda_sessions_ok:
            if require_cuda:
                raise RuntimeError(
                    "VitSAM requires CUDA, but the ONNX Runtime encoder/decoder "
                    "sessions did not both select CUDAExecutionProvider"
                )
            print("VitSam CUDA provider failed to initialize; falling back to CPU")
            self.device = "cpu"
            self.decoder, self.encoder = make_sessions("cpu")

        print(
            "VitSam device:", self.device,
            "encoder providers:", self.encoder.session.get_providers(),
            "decoder providers:", self.decoder.session.get_providers(),
        )

    def __call__(self, img, bboxes):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = self._preprocess(raw_img, img_size=VITSAM_IMAGE_SIZE)
        img_embeddings = self.encoder(img)
        boxes = np.array(bboxes, dtype=np.float32)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )

        return masks, boxes

    def _preprocess(self, x, img_size=VITSAM_IMAGE_SIZE):
        pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

        x = torch.tensor(x)
        resize_transform = SamResize(img_size)
        x = resize_transform(x).float() / 255
        x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

        h, w = x.shape[-2:]
        th, tw = img_size, img_size
        assert th >= h and tw >= w
        x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

        return x
