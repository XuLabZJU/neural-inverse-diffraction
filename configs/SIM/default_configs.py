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
    training.snapshot_freq = 5000  
    training.log_freq = 100        
    training.eval_freq = 1000      
    training.sampling_freq = 1000  
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
    model.sigma = 0.02  
    model.delta = 0.025 
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
    
    model.z_schedule = np.array([model.z_min] + 
    [5.1, 5.7711, 6.468, 7.1921, 7.9446, 8.7268, 9.5402, 10.3862,
      11.2665, 12.1828, 13.137, 14.1309, 15.1668, 16.2468, 17.3733,
     18.5489, 19.7764, 21.0587, 22.3992, 23.8013, 25.2687, 26.8057,
     28.4166, 30.1063, 31.8802, 33.7442, 35.7046, 37.7686, 39.944,
     42.2396, 44.6651, 47.2314, 49.9507, 52.837, 55.906, 59.1756,
     62.6664, 66.4023, 70.4112, 74.7257, 79.3846, 84.4344, 89.9312,
     95.9443, 102.5598, 109.8871, 118.0679, 127.2906, 137.8126, 150])
    
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
