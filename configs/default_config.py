"""
Default configuration for U-RWKV models
"""

class Config:
    # Model parameters
    model_name = "cmunext"
    input_channels = 3
    num_classes = 1
    base_channels = 32
    
    # Training parameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    
    # Dataset parameters
    train_data_path = "data/train"
    val_data_path = "data/val"
    test_data_path = "data/test"
    
    # Optimizer parameters
    optimizer = "AdamW"
    weight_decay = 0.01
    
    # Scheduler parameters
    scheduler = "cosine"
    warmup_epochs = 5
    
    # Augmentation parameters
    use_augmentation = True
    random_flip = True
    random_rotate = True
    random_crop = True
    
    # Hardware parameters
    num_workers = 4
    device = "cuda"
    
    # Logging parameters
    log_interval = 10
    save_interval = 1
    
    # Model specific parameters
    rwkv_hidden_size = 256
    rwkv_num_layers = 4
    rwkv_num_heads = 8 