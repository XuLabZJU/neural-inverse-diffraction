"""All functions and modules related to model definition."""
import torch
import torch.nn as nn
import logging
import numpy as np
from model_code.unet import UNetModel
from PIL import Image
from torchvision import transforms
from ml_collections.config_flags import config_flags
from absl import flags



class AngularSpectrumPropagator(nn.Module):
    def __init__(self, position_zs, config, device):
        """ 
        Initialize microscope-system parameters and the diffraction propagator.

        Responsibilities:
        1. Build the frequency grid used by angular-spectrum propagation.
        2. Model the lens transfer behavior without applying an explicit NA cutoff.
        3. Simulate diffraction with the angular-spectrum method.
        4. Crop the propagated field back to the original object-plane size.
        5. Execute the full forward propagation pipeline.

        Args:
            position_zs: Array of propagation distances, typically derived from
                config.model.z_schedule.
            config: Project configuration object containing optical parameters.
            device: Torch device used for tensor allocation.
        """
        super(AngularSpectrumPropagator, self).__init__()
        
        # Optical field parameters loaded from the config.
        self.wavelength = torch.tensor(float(config.model.wavelength)).to(device)
        self.dx_img = self.dy_img = float(config.model.pixel_size)
        self.W = config.data.image_size
        self.H = config.data.image_size
        self.image_size = config.data.image_size
        self.image_length = (self.W * self.dx_img, self.H * self.dy_img)
        
    # Lens parameters
        self.NA = config.model.NA
        
    # Cutoff frequency for the angular-spectrum formulation
        self.ρc = self.NA/(self.wavelength)
    # Monochromatic field intensity
        self.intensity = torch.tensor(1.0, device=device)
        
    # Propagation distances used by the forward model
        self.position_zs = torch.tensor(position_zs).to(device)
        

    
    def build_freq_grid(self, H, W, dx, dy, device):
        """Build the frequency grid from the current image size and pixel pitch."""
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=dx)).to(device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=dy)).to(device)
        fxx, fyy = torch.meshgrid(fx, fy, indexing="ij")
        return fxx, fyy

    def CTF2OTF(self, CTF):
        # H to h
        apsfde = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(CTF)))
        # PSF
        ipsfde = torch.abs(apsfde)**2
        ipsfde = ipsfde / ipsfde.sum(dim=(-2, -1), keepdim=True)
        # OTF
        OTF = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(ipsfde)))
        return OTF, ipsfde

    def AS(self, x, z, dx, dy):
        """Propagate the field with the angular-spectrum method."""
        H, W = x.shape[-2:]
        fxx, fyy = self.build_freq_grid(H, W, dx, dy, x.device)

        lambda_fx = self.wavelength * fxx
        lambda_fy = self.wavelength * fyy
        sqrt_freq = 1 - lambda_fx**2 - lambda_fy**2

        # Handle both propagating and evanescent components
        valid_sqrt_freq = torch.where(
            sqrt_freq >= 0,
            torch.sqrt(torch.clamp(sqrt_freq, min=0)),   # Propagating waves
            1j * torch.sqrt(torch.abs(sqrt_freq))        # Evanescent waves
        )

        CTF = abs(torch.exp(1j * 2 * torch.pi / self.wavelength * z * valid_sqrt_freq))

        # Incoherent propagation
        OTF, PSF = self.CTF2OTF(CTF)

        fft_I = torch.fft.fftshift(torch.fft.fft2(x))
        I_OTF = torch.fft.ifft2(torch.fft.ifftshift(OTF * fft_I))

        return torch.abs(I_OTF), CTF, OTF, PSF

    def zero_padding(self, x, pad=32):
        """Apply fixed-size zero padding to reduce edge ringing artifacts."""
        padding = (pad, pad, pad, pad)
        return nn.functional.pad(x, padding, mode='constant', value=0)

    def center_crop(self, x, target_size):
        """Crop the propagated field back to the original spatial size."""
        return transforms.CenterCrop(target_size)(x)

    def forward(self, x, fwd_steps: int, pad=32):
        """
        Forward propagation pipeline:
        1. Apply zero padding.
        2. Perform angular-spectrum propagation.
        3. Center-crop back to the model input size.
        """
        # Select the propagation distance for the current forward step.
        if len(x.shape) == 4:  # B,C,H,W
            z = self.position_zs[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:  # C,H,W
            z = self.position_zs[fwd_steps][:, None, None]

        # Build the input field amplitude
        I = self.intensity * x

        # Zero-pad before propagation
        I = self.zero_padding(I, pad=pad)
        # Propagate in free space
        I,_,_,_ = self.AS(I, z, self.dx_img, self.dy_img)
        
        # Crop back to the original image size
        
        I = self.center_crop(I, self.image_size)
        
        intensity = torch.abs(I)  # Intensity magnitude; no extra squaring is required here
        
        return intensity


def create_propagator_from_z_schedule(config, position_zs, device):
    return AngularSpectrumPropagator(position_zs, config, device)


"""The next two functions based on https://github.com/yang-song/score_sde"""


def create_model(config, device_ids=None):
    """Create the model."""
    model = UNetModel(config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    logging.info("Model created with {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model


def get_model_fn(model, train=False):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, fwd_steps):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, fwd_steps)
        else:
            model.train()
            return model(x, fwd_steps)
    return model_fn
