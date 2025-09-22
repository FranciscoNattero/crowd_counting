class P2PNetConfig:
    # P2PNet-specific params
    weight_path = "src/CrowdCounting_P2PNet/weights/SHTechA.pth"   # or your own checkpoint
    backbone = "vgg16_bn"
    row = 2
    line = 2
    score_threshold = 0.5

    # Device
    device_index = 0   # este estoy forzado a ponerlo en 0 pq yolo hace un cuda_visible_devices