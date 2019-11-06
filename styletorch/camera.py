"""Camera

The file contains the Camera module for the live style transfert.
The camera is thread safe to use.
"""
import numpy as np
import threading
import cv2

from typing import Tuple

class Camera:
    """Camera

    Attributes
    ----------
    src        : str
                 Camera source ID
    cap        : cv2.VideoCapture
                 Capture device
    grabbed    : bool
                 Result error for the frame retrieval
    frame      : np.ndarray
                 Current Frame
    stated     : bool
                 Is the Camera currently recording
    read_lock  : threading.Lock
                 Lock to access the current Frame
    subscribers: int
                 Number of instance reading the camera
    """

    def __init__( self: 'Camera', src: int = 0, width: int = 1024, height: int = 576 ) -> None:
        """Init

        Parameters
        ----------
        src   : str
                default 0
                Camera source ID
        width : int
                default 1024
                Width of the captured frame
        height: int
                default 576
                Height of the captured frame

        """
        self.src                 = src
        self.cap                 = cv2.VideoCapture( self.src )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.grabbed, self.frame = self.cap.read( )
        self.started             = False
        self.read_lock           = threading.Lock( )

        self.subscribers         = 0

    def subscribe( self: 'Camera' ) -> None:
        """Subscribe
        """
        with self.read_lock:
            self.subscribers += 1

    def unsubscribe( self: 'Camera' ) -> None:
        """Unsubscribe
        """
        with self.read_lock:
            self.subscribers -= 1

            if self.subscribers <= 0:
                self.stop( )

    def start( self: 'Camera' ) -> 'Camera':
        """Start

        Returns
        -------
        camera: Camera
                Return the camera instance if not already started else None
        """
        if self.started:
            return None

        self.started = True
        self.thread  = threading.Thread( target = self.update, args = ( ) )
        self.thread.start( )
        return self

    def update( self: 'Camera' ) -> None:
        """Update
        """
        while self.started:
            grabbed, frame = self.cap.read( )

            with self.read_lock:
                self.grabbed = grabbed
                self.frame   = frame

    def read( self: 'Camera', flip: bool = True, rgb: bool = True ) -> Tuple[ bool, np.ndarray ]:
        """Read

        Parameters
        ----------
        flip: bool
              Do the frame needs to be flipped
        rgb : bool
              Do the frame needs to be converted to RGB format

        Returns
        -------
        grabbed: bool
                 Has the frame retrieval failed or not
        frame  : np.ndarray
                 Retrieved frame from the camera
        """
        with self.read_lock:
            frame   = self.frame.copy( )

        frame   = cv2.flip( frame, 1 ) if flip else frame
        frame   = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB ) if rgb else frame
        grabbed = self.grabbed

        return grabbed, frame

    def stop( self: 'Camera' ) -> None:
        """Stop
        """
        self.started = False
        self.thread.join( )

    def __exit__( self: 'Camera', exec_type, exc_value, traceback ) -> None:
        """Exit
        """
        self.cap.release( )
