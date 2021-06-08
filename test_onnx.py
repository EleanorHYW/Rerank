import torch
import numpy as np
import onnx
from onnx_model_structure import ptNetEncoder, ptNetDecoder
import onnxruntime as ort
from onnx_export_and_infer import export_onnx, infer_with_onnx


def test(model, model_path, encoder_path, decoder_path):
    export_onnx(model, model_path, encoder_path, decoder_path)
    infer_with_onnx(model, encoder_path, decoder_path)
