import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
    
    def predict(self, image, box_threshold=0.35, text_threshold=0.35):
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
            if score >= box_threshold:
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


class DINO():
    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=False).to(self.device)
        self.classes = None

    def set_classes(self, classes):
        # GroundingDINO expects "." at the end of each query
        self.classes = [cls.lower().strip() + "." for cls in classes]

    def predict(self, image, box_threshold=0.4, text_threshold=0.3):
        if self.classes is None:
            raise ValueError("call set_classes before predict().")

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = Image.fromarray(image_rgb)
        else:
            image_rgb = image

        text_queries = " ".join(self.classes)

        inputs = self.processor(images=image_rgb, text=text_queries, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_rgb.size[::-1]] 
        )[0]

        bboxes, classes, confidences = [], [], []
        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            bboxes.append([xmin, ymin, xmax, ymax])
            classes.append(label_id)  
            confidences.append(float(score))
        return bboxes, classes, confidences
    
    def get_image_with_bboxes(self, image, conf=0.4): 
        bboxes, classes, confidences = self.predict(image, box_threshold=conf) 
        for i in range(len(bboxes)):
             if confidences[i] >= conf: 
                x1, y1, x2, y2 = map(int, bboxes[i]) 
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(image, f"{classes[i]} {confidences[i]:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        return image

class VitSam():

    def __init__(self, encoder_model, decoder_model):
        # GA-15: both arms of this selector read "cpu", directly under a comment saying
        # the device is chosen so the ONNX encoder/decoder use CUDA when available. So
        # every published sam_ms was a CPU ONNX number presented as the system's
        # segmentation cost -- an honest measurement of what the code did, which is why
        # it was invisible in the number alone, and not comparable with the cloud
        # backend's GPU SAM.
        #
        # torch.cuda.is_available() is not sufficient on its own: SamEncoder/SamDecoder
        # pass the choice straight to onnxruntime as providers=["CUDAExecutionProvider"],
        # and torch can see a GPU in a container whose onnxruntime build has no CUDA
        # provider at all. Ask onnxruntime what it actually has.
        import onnxruntime as ort
        _cuda = torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()
        self.device = "cuda" if _cuda else "cpu"
        print("VitSam device:", self.device)

        self.decoder = SamDecoder(decoder_model, device=self.device)
        self.encoder = SamEncoder(encoder_model, device=self.device)


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
