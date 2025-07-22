def select_model(args):
    is_use_wav2vec_ft = False
    if args.model_name == "ASSIST":
        from src.models.ASSIST import AasistEncoder, DownStreamLinearClassifier
        import json
        with open(args.aasist_config_path, "r") as f_json:
            aasist_config = json.load(f_json)
        encoder = AasistEncoder(aasist_config["model_config"]).to(args.device)
        model = DownStreamLinearClassifier(encoder, input_depth=160).to(args.device)
    elif args.model_name == "RawNet2":
        from src.models.RawNet2 import RawNet2
        args.model['nb_classes'] = 2
        model = RawNet2(args.model).to(args.device)
    elif args.model_name == "WAV2VEC_MLP":
        from src.models.MLP import MLPClassifier
        model = MLPClassifier(input_dim=1024, num_classes=2).to(args.device)
        is_use_wav2vec_ft = True
    elif args.model_name == "SLS":
        from src.models.SLS import Model
        import torch.nn as nn
        model = Model(args=None, device=args.device)
        model = nn.DataParallel(model).to(args.device)
        is_use_wav2vec_ft = False
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")
    return model, is_use_wav2vec_ft