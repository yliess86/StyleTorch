"""Inference

The file contains the definition of the Syle Network for inference.
Used only after JIT model conversion.
"""
import numpy as np
import torch
import cv2

from typing import Tuple

class StyleNet:
    """Style Network

    Attributes
    ----------
    model: torch.Module
           Loaded JIT compiled model
    mean : torch.Tensor
           Mean for the dataset
    std  : torch.Tensor
           Std for the dataset
    """

    def __init__( self: 'StyleNet', path: str ) -> None:
        """Init

        Parameters
        ----------
        path: str
              Path to the JIT model
        """
        self.model = torch.jit.load( path )
        self.model = self.model.eval( ).cuda( )

        self.mean  = torch.FloatTensor( [ .485, .456, .406 ] )
        self.mean  = self.mean.view( 3, 1, 1 ).cuda( )
        self.std   = torch.FloatTensor( [ .229, .224, .225 ] )
        self.std   = self.std.view( 3, 1, 1 ).cuda( )

    def __call__( self: 'StyleNet', X: np.ndarray, size: Tuple[ int ] = None ) -> np.ndarray:
        """Call

        Parameters
        ----------
        X   : np.ndarray
              Input image
        size: Tuple[ int ]
              default None
              Output size if image need to be resized

        Returns
        -------
        X: np.ndarray
           Stylized Image
        """
        X = torch.from_numpy( X ).float( )
        X = X.transpose( 0, 2 ).unsqueeze( 0 ).float( ).cuda( )
        X = ( X / 255. - self.mean ) / self.std

        X = ( ( self.model( X ) * self.std + self.mean ) * 255. ).clamp( 0, 255 )
        X = X.squeeze( 0 ).transpose( 0, 2 ).detach( ).cpu( ).numpy( )

        X = X.astype( np.uint8 )
        X = cv2.resize( X, size ) if size is not None else X

        return X
