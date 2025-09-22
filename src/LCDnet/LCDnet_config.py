class LCDNetConfig:
    # data
    images = "path/to/images"           # folder with images
    ann = "path/to/annotations.json"    # JSON with {"images": [{"file_name":..., "width":..., "height":..., "points":[[x,y], ...]}, ...]}
    height = 512
    width = 640

    # training
    batch = 16
    epochs = 100
    lr = 2e-3
    weight_decay = 1e-4
    workers = 8
    warmup_epochs = 3

    # target generation
    ksize = 7
    sigma = 2.0

    # performance
    use_compile = False    
    use_amp = True   
    channels_last = True
    cudnn_benchmark = True
    prefetch_factor = 4

    device = 1