import os
import time

import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder
# H9. The DINO class was deleted (review H9): never instantiated in either tree, and
# its predict appended the raw label_id as a class name -- a defect, not a feature, if it
# ever HAD been wired. The AutoProcessor/AutoModelForZeroShotObjectDetection imports
# went with it: they served nothing else in this file.
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def write_vitsam_status(status):
    """Publish VitSAM startup state to the optional run-level startup gate.

    The launcher and the perception node live in different processes.  A marker in the
    shared run directory lets the host-side Habitat feed wait for the *same* ONNX Runtime
    sessions that will serve real detections, instead of warming a short-lived helper
    process whose CUDA/ORT state could not be reused.
    """
    path = os.environ.get("VITSAM_READY_FILE", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as marker:
            marker.write(f"{status}\n")
        os.replace(temporary, path)
    except OSError as exc:
        # A marker is only a coordination aid. It must never hide the actual model
        # error or make a standalone perception run fail merely because no shared
        # output directory is writable.
        print(f"VitSam status marker unavailable ({path}): {type(exc).__name__}: {exc}")

class OWLv2():
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_id, local_files_only=False)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=False)
        self.model.to(self.device)
        self.model.eval()
        self.classes = None

    
    def set_classes(self, classes):
        # OWL-ViT works with natural language queries
        self.classes = [cls.lower().strip() for cls in classes]
    
    def predict(self, image, box_threshold=0.35):
        # H10. `text_threshold` is DELETED: it was in the signature, used nowhere in the
        # body, and its presence implied a working knob. The box threshold is the one
        # post_process actually applies.
        if self.classes is None:
            raise ValueError("Call set_classes before predict().")
        
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        else:
            image_pil = image
        
        # Prepare text queries
        text_queries = [[f"a photo of a {cls}" for cls in self.classes]]
        
        # Process inputs
        inputs = self.processor(
            text=text_queries, 
            images=image_pil, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]
        
        bboxes, classes, confidences = [], [], []

        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            # H10. The `score >= box_threshold` re-filter is DELETED:
            # post_process_grounded_object_detection was called with threshold=box_threshold
            # and has already applied it -- this loop only RENAMED each survivor.
            xmin, ymin, xmax, ymax = box.cpu().tolist()
            bboxes.append([xmin, ymin, xmax, ymax])
            # --- MODIFICA INIZIO ---
            # Controllo che l'indice restituito dal modello non superi la grandezza della lista
            if int(label) < len(self.classes):
                classes.append(self.classes[int(label)])
            else:
                print(f"[WARN] OWLv2 returned an out-of-range index: {label} (classes available: {len(self.classes)})")
                classes.append("unknown_object") # Assegna un'etichetta di fallback
            # --- MODIFICA FINE ---
            confidences.append(float(score))
        
        return bboxes, classes, confidences
    
    def get_image_with_bboxes(self, image, conf=0.1):
        bboxes, classes, confidences = self.predict(image, box_threshold=conf)
        
        for i in range(len(bboxes)):
            if confidences[i] >= conf:
                x1, y1, x2, y2 = map(int, bboxes[i])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, 
                    f"{classes[i]} {confidences[i]:.2f}", 
                    (x1, max(0, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
        
        return image


class VitSam():

    def __init__(self, encoder_model, decoder_model):
        # VitSAM is an ONNX model. Select the device from ONNX Runtime's providers, not
        # from torch.cuda.is_available(): PyTorch and ONNX Runtime can have different CUDA
        # installations, and the latter is the runtime that executes these two models.
        import onnxruntime as ort

        write_vitsam_status("loading")

        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
        requested_device = "cuda" if cuda_available else "cpu"
        self.device = requested_device

        def make_sessions(device):
            return (
                SamDecoder(decoder_model, device=device),
                SamEncoder(encoder_model, device=device),
            )

        self.decoder, self.encoder = make_sessions(requested_device)

        # ONNX Runtime may list CUDAExecutionProvider but silently fall back to CPU when
        # one of its shared-library dependencies or NVIDIA device permissions is missing.
        # Check the providers selected by both actual sessions so the log never claims GPU
        # execution when VitSAM is really running on CPU.
        actual_providers = set(self.decoder.session.get_providers()) | set(
            self.encoder.session.get_providers())
        if requested_device == "cuda" and "CUDAExecutionProvider" not in actual_providers:
            print("VitSam CUDA provider failed to initialize; falling back to CPU")
            self.device = "cpu"
            self.decoder, self.encoder = make_sessions("cpu")
            actual_providers = set(self.decoder.session.get_providers()) | set(
                self.encoder.session.get_providers())

        print(
            "VitSam device:", self.device,
            "encoder providers:", self.encoder.session.get_providers(),
            "decoder providers:", self.decoder.session.get_providers(),
        )

        # The first ONNX Runtime execution can be substantially slower than steady
        # state because CUDA kernels, execution graphs, and provider memory are
        # initialized lazily.  Pay that cost while the node is starting, rather than
        # blocking the first real perception cycle after the robot has begun moving.
        # The switch is useful for lightweight import/startup tests, but is enabled by
        # default for an actual run.
        if os.environ.get("VITSAM_WARMUP", "1").strip().lower() not in {"0", "false", "no", "off"}:
            if self._warmup():
                write_vitsam_status("warmed")
            else:
                write_vitsam_status("failed")
                if os.environ.get("VITSAM_REQUIRE_WARMUP", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
                    raise RuntimeError(
                        "VitSam warmup failed and VITSAM_REQUIRE_WARMUP is enabled"
                    )
        else:
            print("VitSam warmup disabled by VITSAM_WARMUP")
            write_vitsam_status("warmed-disabled")


    def _warmup(self):
        """Execute one end-to-end synthetic segmentation before live data arrives."""
        warmup_image = np.zeros((256, 256, 3), dtype=np.uint8)
        warmup_bbox = [64.0, 64.0, 192.0, 192.0]
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                masks, _ = self(warmup_image, warmup_bbox)
            elapsed = time.perf_counter() - started
            print(
                f"VitSam warmup inference done: {elapsed:.3f}s "
                f"(device={self.device}, masks_shape={np.asarray(masks).shape})"
            )
            return True
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"VitSam warmup failed after {elapsed:.3f}s: "
                f"{type(exc).__name__}: {exc}. "
                "The first live inference may still pay initialization cost."
            )
            return False


    def __call__(self, img, bboxes):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = self._preprocess(raw_img, img_size=512)
        img_embeddings = self.encoder(img)
        boxes = np.array(bboxes, dtype=np.float32)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )

        return masks, boxes

    def _preprocess(self, x, img_size=512):
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
