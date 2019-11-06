"""Train

The file contains all the code for training.
"""

if __name__ == '__main__':
    import torch.nn as nn
    import numpy as np
    import argparse
    import torch
    import os

    from styletorch.loss import TotalVariationLoss
    from styletorch.download import download_coco
    from torchvision.datasets import ImageFolder
    from styletorch.model import FeaturesNet
    from torch.utils.data import DataLoader
    from styletorch.loss import ContentLoss
    from styletorch.loss import StyleLoss
    from styletorch.model import StyleNet
    from torchvision import transforms
    from torch.optim import Adam
    from tqdm import tqdm
    from PIL import Image

    parser       = argparse.ArgumentParser( )
    parser.add_argument(
        '-e', '--epochs',
        help     = 'Number of Epochs to train the model on',
        type     = int,
        required = True
    )
    parser.add_argument(
        '-b', '--batch_size',
        help    = 'Batch size for training',
        type    = int,
        default = 32
    )
    parser.add_argument(
        '-s', '--size',
        help    = 'Image size',
        type    = int,
        default = 256
    )
    parser.add_argument(
        '-f', '--folder',
        help     = 'Folder of the dataset',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-S', '--style_path',
        help     = 'Style Image path',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-m', '--model_path',
        help     = 'Folder where to save the model',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-T', '--test_path',
        help     = 'Path of the test image',
        type     = str,
        required = True
    )
    parser.add_argument(
        '-d', '--download',
        help     = 'Need to download coco ?',
        action   = 'store_true'
    )
    parser.add_argument(
        '-t', '--t_factor',
        help     = 'T value to prune the model',
        type     = float,
        default  = 1.
    )
    parser.add_argument(
        '-n', '--n_residuals',
        help     = 'Number of residual layers',
        type     = int,
        default  = 5
    )
    parser.add_argument( '--content_alpha',        type = float, default = 1e5  )
    parser.add_argument( '--style_alpha',          type = float, default = 1e10 )
    parser.add_argument( '--totalvariation_alpha', type = float, default = 1e-1 )
    args         = parser.parse_args( )

    if args.download:
        if not os.path.isdir( args.folder ):
            os.makedirs( args.folder )

        download_coco( args.folder )

    epochs       = args.epochs
    batch_size   = args.batch_size
    size         = args.size
    folder       = args.folder
    style_path   = args.style_path
    save_path    = args.model_path
    test_path    = args.test_path

    trans        = transforms.Compose( [
        transforms.Resize( size ),
        transforms.RandomCrop( size ),
        transforms.ToTensor( ),
        transforms.Normalize( ( .485, .456, .406 ), ( .229, .224, .225 ) )
    ] )
    style_trans   = transforms.Compose( [
        transforms.ToTensor( ),
        transforms.Normalize( ( .485, .456, .406 ), ( .229, .224, .225 ) )
    ] )

    mean         = torch.FloatTensor( [ .485, .456, .406 ] ).view( 3, 1, 1 ).cuda( )
    std          = torch.FloatTensor( [ .229, .224, .225 ] ).view( 3, 1, 1 ).cuda( )
    back_trans   = transforms.Compose( [
        transforms.Lambda( lambda X: ( X.squeeze( 0 ) * std + mean ).clamp( 0, 1 ).detach( ).cpu( ) ),
        transforms.ToPILImage( mode = 'RGB' )
    ] )

    dataset      = ImageFolder( folder, trans )
    loader       = DataLoader( dataset, shuffle = True, pin_memory = True, num_workers = 10 )

    stylenet     = nn.DataParallel( StyleNet( args.t_factor, args.n_residuals ) ).cuda( )
    featuresnet  = nn.DataParallel( FeaturesNet( ) ).eval( ).cuda( )

    optim        = Adam( stylenet.parameters( ), lr = 1e-3 )

    content_loss = ContentLoss( args.content_alpha ).cuda( )
    style_loss   = StyleLoss( args.style_alpha ).cuda( )
    totvar_loss  = TotalVariationLoss( args.totalvariation_alpha ).cuda( )

    style_img    = style_trans( Image.open( style_path ).convert( 'RGB' ) ).cuda( )
    style_imgs   = style_img.repeat( batch_size, 1, 1, 1 )
    style_feat   = featuresnet( style_imgs )
    style_grams  = [ style_loss.gram( f ) for f in style_feat ]

    test_img     = Image.open( test_path ).convert( 'RGB' )
    test_img     = style_trans( test_img ).unsqueeze( 0 ).cuda( )

    for epoch in tqdm( range( epochs ), desc = 'Epoch' ):
        pbar     = tqdm( loader, desc = 'Batch' )
        t_loss   = 0.
        tc_loss  = 0.
        ts_loss  = 0.
        ttv_loss = 0.

        for b, batch in enumerate( pbar ):
            optim.zero_grad( )

            content_img, _ = batch
            content_img    = content_img.cuda( )
            content_feat   = featuresnet( content_img )

            generated_img  = stylenet( content_img )
            generated_feat = featuresnet( generated_img )

            c_loss         = content_loss( generated_feat[ 1 ] , content_feat[ 1 ] )
            s_loss         = style_loss( generated_feat, style_grams, content_img.size( 0 ) )
            tv_loss        = totvar_loss( generated_img )
            loss           = c_loss + s_loss + tv_loss


            t_loss        += loss.item( )
            tc_loss       += c_loss.item( )
            ts_loss       += s_loss.item( )
            ttv_loss      += tv_loss.item( )

            loss.backward( )
            optim.step( )

            pbar.set_postfix(
                loss    =   t_loss / ( b + 1 ),
                c_loss  =  tc_loss / ( b + 1 ),
                s_loss  =  ts_loss / ( b + 1 ),
                tv_loss = ttv_loss / ( b + 1 ),
            )

            if b > 0 and b % 200 == 0:
                with torch.no_grad( ):
                    test_generated = stylenet( test_img )
                    test_generated = back_trans( test_generated )
                    test_generated.save( os.path.join(
                        save_path,
                        f'test_{ style_path.split( "/" )[ -1 ].split( "." )[ 0 ] }.png'
                    ) )

                torch.save(
                    { 'model': stylenet.state_dict( ), 't': args.t_factor, 'n': args.n_residuals },
                    os.path.join(
                        save_path,
                        f'model_{ style_path.split( "/" )[ -1 ].split( "." )[ 0 ] }.tar'
                    )
                )
