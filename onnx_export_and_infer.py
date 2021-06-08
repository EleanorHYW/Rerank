import torch
import numpy as np
import onnx
from onnx_model_structure import ptNetEncoder, ptNetDecoder
import onnxruntime as ort


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def turn_model_into_encoder_decoder(model):
    encoder = model.encoder
    decoder = model.decoder
    true_encoder = ptNetEncoder(encoder)
    true_decoder = ptNetDecoder(decoder)
    return true_encoder, true_decoder


def export_onnx(model, model_path, encoder_path, decoder_path):
    model.eval()

    saved_state = torch.load(model_path)
    model.load_state_dict(saved_state['state_dict'])
    encoder, decoder = turn_model_into_encoder_decoder(model)
    # encoder = model.encoder
    # decoder = model.decoder

    batch_size = 8  # 批处理大小
    input_shape = (5, 4)  # 输入数据
    encoder_output_shape = (5, 32)
    # encoder inputs
    n_layers = 1
    hidden_dim = 32
    enc_inp = torch.randn(batch_size, *input_shape)
    enc_h0 = torch.randn(n_layers, batch_size, hidden_dim)
    enc_c0 = torch.randn(n_layers, batch_size, hidden_dim)

    # decoder_inputs
    dec_inp = enc_inp
    dec_his = torch.randn(batch_size, *input_shape)
    dec_inp0 = torch.mean(dec_his, dim=1)
    dec_h0 = torch.randn(batch_size, hidden_dim)
    dec_c0 = torch.randn(batch_size, hidden_dim)
    enc_out = torch.randn(batch_size, *encoder_output_shape)

    # !Careful: set export_params to False can make the exported model first take all of its parameters as arguments
    # the ordering as specified by model.state_dict().values()
    with torch.no_grad():
        torch_out, _ = model(dec_inp, dec_his)
        torch.onnx.export(
            decoder,
            (dec_inp, dec_inp0, dec_h0, dec_c0, enc_out),
            decoder_path,
            export_params=True,
            verbose=True,
            input_names=['embedded_inputs', 'dec_input', 'dec_h0', 'dec_c0', 'enc_outs'],
            output_names=['dec_output', 'pointer', 'dec_hn', 'dec_cn'],
            do_constant_folding=False,
            dynamic_axes={
                'embedded_inputs': {0: 'batch_size', 1: 'sequence'},
                'dec_input': {0: 'batch_size'},
                'dec_h0': {0: 'batch_size'},
                'dec_c0': {0: 'batch_size'},
                'enc_outs': {0: 'batch', 1: 'sequence'},
            },
            opset_version=11,
        )
        enc_h0 = None
        enc_c0 = None
        torch.onnx.export(
            encoder,
            (enc_inp, enc_h0, enc_c0),
            encoder_path,
            export_params=True,
            verbose=True,
            input_names=['embedded_inputs', 'enc_h0', 'enc_c0'],
            output_names=['enc_outputs', 'enc_hn', 'enc_cn'],
            do_constant_folding=False,
            dynamic_axes={
                'embedded_inputs': {0: 'batch', 1: 'sequence'},
                'enc_h0': {0: 'batch', 1: 'sequence'},
                'enc_c0': {0: 'batch', 1: 'sequence'},
            },
            opset_version=12,
        )
        print("Onnx export succeed!")

        print("Test!")
        ort_session1 = ort.InferenceSession("./" + encoder_path)
        # compute ONNX Runtime output prediction
        encoder_inputs = {ort_session1.get_inputs()[0].name: to_numpy(enc_inp),
                          # ort_session1.get_inputs()[1].name: enc_h0,
                          # ort_session1.get_inputs()[2].name: enc_c0,
                          }
        encoder_outs, encoder_hn, encoder_cn = ort_session1.run(None, encoder_inputs)
        import pdb;
        pdb.set_trace()
        dec_h0 = encoder_hn[-1]
        dec_c0 = encoder_cn[-1]
        ort_session2 = ort.InferenceSession("./" + decoder_path)
        decoder_inputs = {ort_session2.get_inputs()[0].name: to_numpy(dec_inp),
                          ort_session2.get_inputs()[1].name: to_numpy(dec_inp0),
                          ort_session2.get_inputs()[2].name: dec_h0,
                          ort_session2.get_inputs()[3].name: dec_c0,
                          ort_session2.get_inputs()[4].name: encoder_outs,
                          }
        dec_output, pointer, dec_hn, dec_cn = ort_session2.run(None, decoder_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), dec_output, rtol=1e-03, atol=1e-05)

        return True


def infer_with_onnx(model, encoder_path, decoder_path):
    _onnx_model_enc = onnx.load("./" + encoder_path)
    _onnx_model_dec = onnx.load("./" + decoder_path)
    print(_onnx_model_enc.graph)
    print(_onnx_model_dec.graph)
    onnx.checker.check_model(_onnx_model_enc)
    onnx.checker.check_model(_onnx_model_dec)
    onnx.helper.printable_graph(_onnx_model_enc.graph)
    onnx.helper.printable_graph(_onnx_model_dec.graph)

    rerank_list_dynamic = torch.rand(8, 12, 4)
    h0 = torch.randn(1, 8, 32)
    c0 = torch.randn(1, 8, 32)
    history_dynamic = torch.rand(8, 6, 4)
    dec_inp0 = torch.mean(history_dynamic, dim=1)

    with torch.no_grad():
        torch_out, _ = model(rerank_list_dynamic, history_dynamic)

    ort_session1 = ort.InferenceSession("./" + encoder_path)
    # compute ONNX Runtime output prediction
    encoder_inputs = {ort_session1.get_inputs()[0].name: to_numpy(rerank_list_dynamic),
                      # ort_session1.get_inputs()[1].name: enc_h0,
                      # ort_session1.get_inputs()[2].name: enc_c0,
                      }
    encoder_outs, encoder_hn, encoder_cn = ort_session1.run(None, encoder_inputs)

    dec_h0 = encoder_hn[-1]
    dec_c0 = encoder_cn[-1]
    ort_session2 = ort.InferenceSession("./" + decoder_path)
    decoder_inputs = {ort_session2.get_inputs()[0].name: to_numpy(rerank_list_dynamic),
                      ort_session2.get_inputs()[1].name: to_numpy(dec_inp0),
                      ort_session2.get_inputs()[2].name: dec_h0,
                      ort_session2.get_inputs()[3].name: dec_c0,
                      ort_session2.get_inputs()[4].name: encoder_outs,
                      }
    dec_output, pointer, dec_hn, dec_cn = ort_session2.run(None, decoder_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), dec_output, rtol=1e-03, atol=1e-05)
