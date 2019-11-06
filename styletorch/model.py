"""Model

The file contains all the models used for style transfert:
    * The Style Network
    * The Feature Network
"""
import torch.nn as nn
import numpy as np
import torch

from styletorch.modules import ResidualLayer
from styletorch.modules import ConvLayer
from styletorch.modules import DeconvLayer
from torchvision import models
from typing import Tuple

class StyleNet( nn.Module ):
    """Style Network

    Attributes
    ----------
    layers: torch.nn.Module
            Sequential Module containing all the module for the style network.
    """

    def __init__( self: 'StyleNet', t: float = 1., n: int = 5 ) -> None:
        """Init

        Parameters
        ----------
        t: float
           Interpolation factor to reduce the model size
        n: int
           Number of residual layers
        """
        super( StyleNet, self ).__init__( )
        self.layers = nn.Sequential(
            ConvLayer(                         3, int( np.floor( t *  32 ) ), 9, 1 ), nn.ReLU( ),
            ConvLayer( int( np.floor( t * 32 ) ), int( np.floor( t *  64 ) ), 3, 2 ), nn.ReLU( ),
            ConvLayer( int( np.floor( t * 64 ) ), int( np.floor( t * 128 ) ), 3, 2 ), nn.ReLU( ),
            *( [ ResidualLayer( int( np.floor( t * 128 ) ), 3 ) for _ in range( n ) ] ),
            DeconvLayer( int( np.floor( t * 128 ) ), int( np.floor( t * 64 ) ), 3, 1, 2 ), nn.ReLU( ),
            DeconvLayer( int( np.floor( t *  64 ) ), int( np.floor( t * 32 ) ), 3, 1, 2 ), nn.ReLU( ),
            ConvLayer( int( np.floor( t * 32 ) ), 3, 9, 1, norm = 'None' )
        )

    def forward( self: 'StyleNet', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input tensor ( image )

        Returns
        -------
        X: torch.Tensor
           Output Tensor ( stylized image )
        """
        return self.layers( X )

class FeaturesNet( nn.Module ):
    """Features Network

    Attributes
    ----------
    slice1: torch.nn.Module
            First slice of the VGG16 model
    slice2: torch.nn.Module
            Second slice of the VGG16 model
    slice3: torch.nn.Module
            Third slice of the VGG16 model
    slice4: torch.nn.Module
            Fourth slice of the VGG16 model
    """

    def __init__( self: 'FeaturesNet' ) -> None:
        """Init
        """
        super( FeaturesNet, self ).__init__( )
        vgg16         = models.vgg16( pretrained = True )
        features      = vgg16.features
        self.slice1   = torch.nn.Sequential( *[ features[ i ] for i in range( 4 ) ] )
        self.slice2   = torch.nn.Sequential( *[ features[ i ] for i in range(  4,  9 ) ] )
        self.slice3   = torch.nn.Sequential( *[ features[ i ] for i in range(  9, 16 ) ] )
        self.slice4   = torch.nn.Sequential( *[ features[ i ] for i in range( 16, 23 ) ] )

        for param in self.parameters( ):
            param.requires_grad = False

    def forward( self: 'FeaturesNet', X: torch.Tensor ) -> Tuple[ torch.Tensor ]:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input Tensor to get the features for

        Returns
        -------
        relu1_2: torch.Tensor
                 Feature Maps from the VGG16 relu1_2
        relu2_2: torch.Tensor
                 Feature Maps from the VGG16 relu2_2
        relu3_3: torch.Tensor
                 Feature Maps from the VGG16 relu3_3
        relu4_3: torch.Tensor
                 Feature Maps from the VGG16 relu4_3
        """
        X        = self.slice1( X )
        relu1_2  = X
        X        = self.slice2( X )
        relu2_2  = X
        X        = self.slice3( X )
        relu3_3  = X
        X        = self.slice4( X )
        relu4_3  = X

        return relu1_2, relu2_2, relu3_3, relu4_3
