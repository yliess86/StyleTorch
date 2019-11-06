"""JIT

The file contains all the code to convert a trained model into a quantized jit
compiled model.
"""
if __name__ == '__main__':
    import torch.quantization
    import argparse
    import torch

    from styletorch.model import StyleNet
    from collections import OrderedDict

    parser          = argparse.ArgumentParser( )
    parser.add_argument(
        '-s', '--source',
        help     = 'Source path to the model',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-d', '--destination',
        help     = 'Destination path to the jit model',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-j', '--jit',
        help   = 'Jit',
        action = 'store_true'
    )
    parser.add_argument(
        '-q', '--quantized',
        help   = 'Quantize',
        action = 'store_true'
    )
    args            = parser.parse_args( )

    model_path      = args.source
    jit_model_path  = args.destination

    new_state_dict  = OrderedDict()
    state_dict      = torch.load( model_path, map_location = 'cuda:0' )
    t               = state_dict[ 't' ]
    n               = state_dict[ 'n' ]

    for k, v in state_dict[ 'model' ].items():
        new_state_dict[ k[ 7: ] ] = v

    model           = StyleNet( t, n )
    model.load_state_dict( new_state_dict )
    model           = model.eval( ).cuda( )

    if not args.jit and not args.quantized:
        torch.save( { 'model': model.state_dict( ), 't': t }, jit_model_path )

    elif args.jit or args.quantized:
        in_random       = torch.rand( ( 1, 3, 512, 512 ) , dtype = torch.float32 ).cuda( )
        jit_model       = torch.jit.trace( model, in_random )

        if args.quantized:
            jit_model = torch.quantization.convert( jit_model )

        torch.jit.save( jit_model, jit_model_path )

    else:
        pass
