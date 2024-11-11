import torch.onnx
import argparse
import os
import json
from torch.utils.mobile_optimizer import optimize_for_mobile
from models import get_net_by_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to model dir', type=str, default=None)
    args = parser.parse_args()

    # Load your trained PyTorch model checkpoint
    model_config_path = '%s/net.config' % args.path
    #trained_weights = args.ckp
    best_model_path = '%s/checkpoint/checkpoint.pth.tar' % args.path

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile(model_config_path):
        # load net from file
        net_config = json.load(open(model_config_path, 'r'))
        model = get_net_by_name(net_config['name']).build_from_config(net_config)
        #model = torch.nn.DataParallel(model)
        model.to(device)
        print(model)

    checkpoint = torch.load(best_model_path, map_location=device)
    if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    
    model.load_state_dict(checkpoint)

    model.eval()

    # # # Dummy input data for the model
    #dummy_input = torch.randn(1, 1, 10, 414).cuda()  # Replace with your input shape
    dummy_input = torch.randn(1, 44100).cuda()  # Batch size of 1, 1 channel, and audio length of 16000
    #torchscript_model = torch.jit.trace(model, dummy_input)

    #torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    #torch.jit.save(torchscript_model_optimized, "/home/majam001/kws/alpha-kws/models/model_f1score_mfcclayer/model_f1score_mfcclayer_torchscript.pt")
    #torchscript_model_optimized._save_for_lite_interpreter("/home/majam001/kws/alpha-kws/models/model_f1score/model_f1score_mfcclayer_torchscript_lite.ptl")
    # # # Export the model to ONNX format
    onnx_path = "/home/majam001/kws/alpha-kws/models/pless_sweep_mfcc40_5/model_3/onnx/model_3_pless_sweep_convmel.onnx"
    #torch.onnx.export(model, dummy_input, onnx_path, keep_initializers_as_inputs=True, do_constant_folding=False, input_names = ['input'], output_names=['output'], verbose=True, dynamic_axes={'input': {2: 'audio_length'}}, opset_version=11)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False, opset_version=11)