"""Main

The file contains a use case of the style transfer pipeline using cameras feed
and jit compiled trained models.
"""
import screeninfo
import argparse
import os

from styletorch.inference import StyleNet
from styletorch.camera import Camera
from glumpy import app, gl, gloo

dirname  = os.path.dirname( __file__ )
vertex   = os.path.join( dirname , 'res/texture.vs' )
fragment = os.path.join( dirname , 'res/texture.fs' )

with open( os.path.abspath(   vertex ), 'r' ) as f: vertex   = f.read( )
with open( os.path.abspath( fragment ), 'r' ) as f: fragment = f.read( )

parser    = argparse.ArgumentParser( )

parser.add_argument( '-m', '--models',  help = 'Models source path', type = str, nargs = '+', required = True           )
parser.add_argument( '-c', '--cameras', help = 'Camera ids',         type = str, nargs = '+', required = True           )
parser.add_argument( '-i', '--input',   help = 'Input camera size',  type = int, nargs = '+', default  = [ 1024,  576 ] )
parser.add_argument( '-o', '--output',  help = 'Output window size', type = int, nargs = '+', default  = [ 1920, 1080 ] )

args      = parser.parse_args( )
cw, ch    = args.input
ww, wh    = args.output
s_info    = screeninfo.get_monitors( )

models    = { path.split( '/' )[ -1 ].split( '.' )[ 0 ]: StyleNet( path ) for path in set( args.models ) }
cameras   = { idx: Camera( int( idx ), cw, ch ) for idx in set( args.cameras ) }

quads     = [ gloo.Program( vertex, fragment, count = 4 ) for _ in range( len( args.models ) ) ]
screens   = [ s_info[ idx % len( s_info ) ] for idx in range( len( args.models ) ) ]
windows   = [ app.Window( width = ww, height = wh, title = str( idx ), fullscreen = True ) for idx in range( len( args.models ) ) ]

for i in range( len( args.models ) ):
    code = f'''
path             = args.models[ i ]
cam              = args.cameras[ i ]
quad             = quads[ i ]
window{i}        = windows[ i ]
screen           = screens[ i ]
window{i}.set_position( screen.x - 1, screen.y - 1 )
window{i}.model  = models[ path.split( '/' )[ -1 ].split( '.' )[ 0 ] ]
window{i}.camera = cameras[ cam ]
window{i}.quad   = quad

@window{i}.event
def on_init( ):
    window{i}.quad[ 'position' ] = [ ( -1, -1 ), ( -1, 1 ), ( 1, -1 ), ( 1, 1 ) ]
    window{i}.quad[ 'texcoord' ] = [ (  0,  1 ), (  0, 0 ), ( 1,  1 ), ( 1, 0 ) ]

    window{i}.camera.subscribe( )
    window{i}.camera.start( )

@window{i}.event
def on_draw( dt ):
    _, capture                  = window{i}.camera.read( )
    window{i}.quad[ 'texture' ] = window{i}.model( capture, ( window{i}.width, window{i}.height ) )

    window{i}.clear( )
    window{i}.quad.draw( gl.GL_TRIANGLE_STRIP )

@window{i}.event
def on_exit( ):
    window{i}.camera.unsubscribe( )
    '''
    exec( code.format( i ) )

app.use( 'pyglet' )
app.run( framerate = 30 )
