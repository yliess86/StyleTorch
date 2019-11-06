"""Modules

The file contains all the pytorch modules needed for the model creation.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvLayer( nn.Module ):
    """Convolution Layer

    Attributes
    ----------
    reflect_pad: int
                 Reflection padding size
    conv       : torch.nn.Module
                 Convolutional Module
    norm       : torch.nn.Module
                 default 'instance'
                 Normalization Module ( InstanceNorm2d or BatchNorm2d )
    """

    def __init__( self: 'ConvLayer', in_c: int, out_c: int, kernel_size: int, stride: int, norm: str = 'instance' ) -> None:
        """Init

        Parameters
        ----------
        in_c       : int
                     Input Channels
        out_c      : int
                     Output Channels
        kernel_size: int
                     Kernel size for the Convolution
        stride     : int
                     Stride for the Convolution
        norm       : str
                     default 'instance'
                     Normalization Module ( 'instance' or 'batch' or None )
        """
        super( ConvLayer, self ).__init__( )
        pad_size         = kernel_size // 2
        self.reflect_pad = nn.ReflectionPad2d( pad_size )
        self.conv        = nn.Conv2d( in_c, out_c, kernel_size, stride )
        self.norm        = nn.InstanceNorm2d( out_c, affine = True ) if norm == 'instance' else \
                           nn.BatchNorm2d( out_c, affine = True ) if norm == 'batch' else \
                           None

    def forward( self: 'ConvLayer', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input tensor given to the convolution.

        Returns
        -------
        X: torch.Tensor
           Convolved tensor output
        """
        X = self.conv( self.reflect_pad( X ) )
        X = X if self.norm is None else self.norm( X )
        return X

class ResidualLayer( nn.Module ):
    """Residual Layer

    Attributes
    ----------
    conv1: torch.nn.Module
           First Convolutional Layer
    conv2: torch.nn.Module
           Second Convolutional Layer
    """

    def __init__( self: 'ResidualLayer', channels: int = 128, kernel_size: int = 3 ) -> None:
        """Init

        Parameters
        ----------
        channels   : int
                     default 128
                     Number of channels for the Convolutions
        kernel_size: int
                     default 3
                     Kernel size for the Convolutions
        """
        super( ResidualLayer, self ).__init__( )
        self.conv1 = ConvLayer( channels, channels, kernel_size, stride = 1 )
        self.conv2 = ConvLayer( channels, channels, kernel_size, stride = 1 )

    def forward( self: 'ResidualLayer', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input tensor given to the convolutional residual layer

        Returns
        -------
        X: torch.Tensor
           Residualy convolved tensor output
        """
        identity = X
        X        = torch.relu( self.conv1( X ) )
        X        = self.conv2( X )
        X        += identity
        return X

class DeconvLayer( nn.Module ):
    """Deconvolution Layer

    Attributes
    ----------
    reflect_pad: int
                 Reflection padding size
    upsample   : int
                 default None
                 Upsampling factor
    conv       : torch.nn.Module
                 Deconvolutional Module
    norm       : torch.nn.Module
                 default 'instance'
                 Normalization Module ( InstanceNorm2d or BatchNorm2d )
    """

    def __init__( self: 'DeconvLayer', in_c: int, out_c: int, kernel_size: int, stride: int, upsample = None, norm: str = 'instance' ) -> None:
        """Init

        Parameters
        ----------
        in_c       : int
                     Input Channels
        out_c      : int
                     Output Channels
        kernel_size: int
                     Kernel size for the Deconvolution
        stride     : int
                     Stride for the Deconvolution
        upsample   : int
                     default None
                     Upsampling factor
        norm       : str
                     default 'instance'
                     Normalization Module ( 'instance' or 'batch' or None )
        """
        super( DeconvLayer, self ).__init__( )
        pad_size         = kernel_size // 2
        self.reflect_pad = nn.ReflectionPad2d( pad_size )
        self.upsample    = upsample
        self.conv        = nn.Conv2d( in_c, out_c, kernel_size, stride )
        self.norm        = nn.InstanceNorm2d( out_c, affine = True ) if norm == 'instance' else \
                           nn.BatchNorm2d( out_c, affine = True ) if norm == 'batch' else \
                           None

    def forward( self: 'DeconvLayer', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input tensor given to the deconvolution.

        Returns
        -------
        X: torch.Tensor
           Deconvolved tensor output
        """
        X = X if self.upsample is None else F.interpolate( X, mode = 'nearest', scale_factor = self.upsample )
        X = self.conv( self.reflect_pad( X ) )
        X = X if self.norm is None else self.norm( X )
        return X
