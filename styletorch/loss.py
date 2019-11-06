"""Loss

The file contains all the loss modules used to train the style transfert network:
    * Content Loss to mesure the amount of content preserved from the content image
    * Style Loss to mesure the amount of style preserved from the style image
    * Total Variation Loss to mesure the pixel proximity relation in the generated image
"""
import torch.nn as nn
import numpy as np
import torch

class ContentLoss( nn.Module ):
    """Content Loss

    Attributes
    ----------
    mse  : torch.nn.Module
           MSE loss module
    alpha: float
           Weight for the loss
    """

    def __init__( self: 'ContentLoss', alpha: float ) -> None:
        """Init

        Parameters
        ----------
        alpha: float
               Weight for the loss
        """
        super( ContentLoss, self ).__init__( )
        self.mse   = nn.MSELoss( )
        self.alpha = alpha

    def forward( self: 'ContentLoss', gen: torch.Tensor, con: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        gen: torch.Tensor
             Generated image
        con: torch.Tensor
             Content image

        Returns
        -------
        loss: torch.Tensor
              Content Loss
        """
        return self.alpha * self.mse( gen, con )

class StyleLoss( nn.Module ):
    """Style Loss

    Attributes
    ----------
    mse  : torch.nn.Module
           MSE loss module
    alpha: float
           Weight for the loss
    """

    def __init__( self: 'StyleLoss', alpha: float ) -> None:
        """Init

        Parameters
        ----------
        alpha: float
               Weight for the loss
        """
        super( StyleLoss, self ).__init__( )
        self.mse   = nn.MSELoss( )
        self.alpha = alpha

    @classmethod
    def gram( self: 'StyleLoss', X: torch.Tensor ) -> torch.Tensor:
        """Gram Matrix

        Parameters
        ----------
        X: torch.Tensor
           Input Tensor

        Returns
        -------
        gram: torch.FloatTensor
              Gram Matrix from the Input Tensor
        """
        B, C, H, W = X.size( )
        N          = np.float32( C * H * W )
        X          = X.view( B, C, H * W )
        return torch.matmul( X, X.transpose( 1, 2 ) ) / N

    def forward( self: 'StyleLoss', gen: torch.Tensor, style_grams: torch.Tensor, batch_size: int ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        gen        : torch.Tensor
                     Generated image
        style_grams: torch.Tensor
                     Gram matrices from the style image
        batch_size : int
                     Batch size

        Returns
        -------
        loss: torch.Tensor
              Style Loss
        """
        loss = 0.
        for gen_feat, style_gram in zip( gen, style_grams ):
            loss += self.mse( self.gram( gen_feat ), style_gram[ :batch_size, :, : ] )
        return self.alpha * loss

class TotalVariationLoss( nn.Module ):
    """Total Varation Loss

    Attributes
    ----------
    alpha: float
           Weight for the loss
    """

    def __init__( self: 'TotalVariationLoss', alpha: float ) -> None:
        """Init

        Parameters
        ----------
        alpha: float
               Weight for the loss
        """
        super( TotalVariationLoss, self ).__init__( )
        self.alpha = alpha

    def forward( self: 'TotalVariationLoss', X: torch.Tensor ) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Generated image

        Returns
        -------
        loss: torch.Tensor
              Content Loss
        """
        return self.alpha * ( torch.sum( torch.abs(
            X[ :, :, :,  :-1 ] - \
            X[ :, :, :, 1:   ] )
        ) + torch.sum( torch.abs(
            X[ :, :,  :-1, : ] - \
            X[ :, :, 1:  , : ]
        ) ) )
