from .model_node import *
import torch

model_dict = {
    "model_for_node_LoL3": model_for_node_LoL3,
}

def freeze_layer(model, config):
    for p1 in config:
        used = False
        for name, param in model.named_parameters():
            if p1 in name:
                param.requires_grad_(False)
                used = True
        assert used, 'Unrecognized parameter name: %s' % p1
    return model


def build_model(config, ckpt_path=None):

    model_name = config.CLIP.model_name
    if model_name in ["P2S_CLIP_Node_Att_qkv", "P2S_CLIP_Node_Att_qkv_2"]:
        model = model_dict[model_name](config)
    else:
        model = model_dict[model_name](config.CLIP)
    config = config.CLIP

    if ckpt_path is not None: # load pre-trained, for eval
        state_dict = torch.load(ckpt_path)['state']
        if 'module' in [k for k in state_dict.keys()][0]:
            state_dict = {k[7:]:v for k,v in state_dict.items()} # e.g., "module.text_net.transformer.resblocks.11.ln_2.weight" -> "text_net.transformer.resblocks.11.ln_2.weight"
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # assert len(unexpected_keys) == 0, 
        print(missing_keys, unexpected_keys)
        print("Load ckpt from %s" % ckpt_path)
        
    else: # load pre-trained clip
        if not hasattr(model, "load_checkpoint"):
            print("model has no attr 'load_checkpoint', generally load it...")
            if config.get('initial', False):
                state_dict = torch.load(config.initial.weight, map_location='cpu')
                # state_dict = torch.load(config.initial.weight)
                if 'state' in state_dict.keys():  
                    state_dict = state_dict['state']
                if 'module' in [k for k in state_dict.keys()][0]:
                    state_dict = {k[7:]:v for k,v in state_dict.items()} # e.g., "module.text_net.transformer.resblocks.11.ln_2.weight" -> "text_net.transformer.resblocks.11.ln_2.weight"
                # state_dict = {k:v for k,v in state_dict.items() if 'logit_scale' not in k}
                if not config.initial.get("load_text_pretrain", True): # not load pretrained text encoder
                    state_dict = {k:v for k,v in state_dict.items() if 'text_net' not in k}
                if not config.initial.get("load_visual_pretrain", True): # not load pretrained visual encoder
                    state_dict = {k:v for k,v in state_dict.items() if 'visual_net' not in k}
                if config.visual.get('resolution', 224) != 224:
                    state_dict = {k:v for k,v in state_dict.items() if 'visual_net.conv1' and 'visual_net.positional_embedding' not in k}
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print("missing_keys", missing_keys)
                print("unexpected_keys", unexpected_keys)
                # assert len(unexpected_keys) == 0

            if config.get("freeze", False):
                # model = freeze_layer(model, config.freeze)
                if config.freeze.get("freeze_text_encoder", False):
                    print("###### Setting: freeze_text_encoder. ######")
                    for name, param in model.named_parameters():
                        if "text_net" in name:
                            param.requires_grad_(False)
                if config.freeze.get("freeze_visual_encoder", False):
                    print("###### Setting: freeze_visual_encoder. ######")
                    for name, param in model.named_parameters():
                        if "visual_net" in name:
                            param.requires_grad_(False)

            if config.get("finetune", False):
                for name, param in model.named_parameters():
                    if "text_net" in name or "visual_net" in name:
                        param.requires_grad_(False)
                print([name for name, param in model.named_parameters() if param.requires_grad])

    return model
    
