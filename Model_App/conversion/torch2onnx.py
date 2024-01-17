import torch
import torch.nn as nn
from model import create_model
from config import *
import onnx

if __name__ == '__main__':
    """
    model = create_model()
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
    model.eval()


    example_input = torch.randn(1, 3, 224, 224)

    # 모델을 ONNX 포맷으로 변환
    torch.onnx.export(model,               
                    example_input,      
                    "model.onnx",        
                    export_params=True, 
                    opset_version=10,  
                    do_constant_folding=True,
                    input_names = ['input'],  
                    output_names = ['output'], 
                    dynamic_axes={'modelInput' : {0 : 'batch_size'}, 
                                    'modelOutput' : {0 : 'batch_size'}})
    """
    
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
