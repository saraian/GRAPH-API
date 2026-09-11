import os

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
