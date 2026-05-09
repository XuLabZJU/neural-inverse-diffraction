import ml_collections
import torch
import numpy as np


def get_config():
    return get_default_configs()


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 1
    training.n_iters = 1300001
    training.snapshot_freq = 5000 # 50000
    training.log_freq = 50         # 50
    training.eval_freq = 1000      # Frequency for reporting loss
    training.sampling_freq = 1000  # Frequency for generating reconstruction samples.
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1
    evaluate.enable_sampling = False
    evaluate.num_samples = 5000 # 50000 
    evaluate.enable_loss = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'BioSR'
    data.image_size = 512
    data.random_flip = True
    data.random_enhance = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1

    # model
    config.model = model = ml_collections.ConfigDict()
    model.K = 50
    model.sigma = 0.02   # Noise level used during training.
    model.delta = 0.025 # Noise level used during sampling.
    model.dropout = 0.1
    model.model_channels = 128  # Base amount of channels in the model
    model.channel_mult = (1, 2, 2)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = (2,)
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.num_res_blocks = 4
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True
    model.loss_norm = False
    
    model.NA = 1.35
    model.wavelength = 500
    model.pixel_size = 31.3
    model.z_max = model.wavelength * 0.3
    model.z_min = 0.01 * model.wavelength
    
    model.z_schedule = np.array([0]+
        np.geomspace(model.z_min, model.z_max, model.K).tolist())
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4 # 2e-4 → 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 0.3
    optim.automatic_mp = False

    config.seed = 42
    config.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
