import onnx
import numpy as np
from onnx_tf.backend import prepare
from object_detector.object_detection import ObjectDetection
import shapely
import pandas as pd
import time
from object_detector.utils import create_box, create_center_point, decode_qr_codes
import json
import re


class QRModel(ObjectDetection):
    """Object Detection class for CNTK
    """

    def __init__(self, model_path, labels=["pair", "qr"], threshold=0.3):
        super(QRModel, self).__init__(labels)
        onnx_model = onnx.load_model(model_path)
        onnx.checker.check_model(onnx_model)
        self.model = prepare(onnx_model)
        self.threshold = threshold
        # should be a regex once we the location code formats
        self.location_regex = "\w{2}\d{5}"

    def predict(self, image):
        """Takes an image and returns a list of predictions"""
        inputs = self.preprocess(image)
        prediction_outputs = self._predict(inputs)
        return self.postprocess(prediction_outputs)

    def is_location_code(self, code):
        return re.match(self.location_regex, code) != None

    def _predict(self, image):
        outputs = self.model.run(image).model_outputs0
        outputs = np.squeeze(outputs).transpose((1, 2, 0))
        return outputs

    def predict_batch(self, images):
        arrays = []
        for image in images:
            image_array = self.preprocess(image)
            arrays.append(image_array)

        arrays = np.concatenate(arrays, axis=0)
        outputs = self.model.run(arrays)
        outputs = outputs.model_outputs0

        results = []

        for i in range(len(arrays)):
            output = outputs[i, :, :, :]
            output = output.transpose((1, 2, 0))
            result = self.postprocess(output)
            results.append(result)

        return results

    def create_dict_batch(self, preds, images):

        l = []

        for i in range(len(preds)):
            d = self.create_dict(preds[i], images[i])
            l.extend(d)

        return l

    def create_dict(self, preds, image):
        """Takes the predictions and a image and returns a dict maping location to product
        and also decodes the qr codes"""

        preds = [d for d in preds if d["probability"] > self.threshold]
        pairs = [d for d in preds if d["tagName"] == "pair"]
        qr_codes = [d for d in preds if d["tagName"] == "qr"]

        for pair in pairs:
            bb = self.widen_box(pair["boundingBox"])
            pair["box"] = create_box(bb)

        decoded_qr_codes = decode_qr_codes(image, qr_codes)
        # Change these to use regex
        locations = [
            d for d in decoded_qr_codes if self.is_location_code(d["code"])]
        products = [
            d for d in decoded_qr_codes if not self.is_location_code(d["code"])]

        df = pd.DataFrame.from_dict(
            [{"location": location["code"], "productId": None, "time": pd.Timestamp.now()} for location in locations])

        for pair in pairs:
            for location in locations:
                for product in products:
                    if pair["box"].contains(product["point"]) and pair["box"].contains(location["point"]):
                        i = df.location == location["code"]
                        df.loc[i, "productId"] = product["code"]
                        # currently each can location can only have a single product hence the break
                        break

        return df.to_dict('records')
    
    def widen_box(self,bb):

        n = 1.5
        bb = bb.copy()

        bb["left"] = max(0, bb["left"] - bb["left"]/ 9 )
        bb["top"] = max(0, bb["top"] - bb["left"] / 9)
        bb["width"] = min(bb["width"] * n, 1)
        bb["height"] = min(bb["height"] * n, 1)
        return bb

#     def widen_box(self, bb):

#         n = 1.5

#         bb["left"] = max(0, bb["left"] + bb["left"] / 7)
#         bb["top"] = max(0, bb["top"] + bb["top"] / 7)
#         bb["width"] = min(bb["width"] * n, 1)
#         bb["height"] = min(bb["height"] * n, 1)
#         return bb
