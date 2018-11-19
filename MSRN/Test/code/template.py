def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.lr_decay = 500
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.lr_decay = 150

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 5
        args.n_resblocks = 10
        args.n_feats = 64
        args.chop = True

    if args.template.find('BMBN') >= 0:
        args.model = 'BMBN'
        args.n_bmbblocks = 40
        args.n_feats = 64
        args.chop = True

    if args.template.find('STHRN_A') >= 0:
        args.model = 'STHRN_A'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_B') >= 0:
        args.model = 'STHRN_B'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_C') >= 0:
        args.model = 'STHRN_C'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_D') >= 0:
        args.model = 'STHRN_D'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_E') >= 0:
        args.model = 'STHRN_E'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_F') >= 0:
        args.model = 'STHRN_F'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_G') >= 0:
        args.model = 'STHRN_G'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_H') >= 0:
        args.model = 'STHRN_H'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_I') >= 0:
        args.model = 'STHRN_I'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('STHRN_J') >= 0:
        args.model = 'STHRN_J'
        args.n_bmbblocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('RTDN') >= 0:
        args.model = 'RTDN'
        args.n_blocks = 5
        args.n_feats = 32
        args.chop = True

    if args.template.find('RTDN_CVPR') >= 0:
        args.model = 'RTDN_CVPR'
        args.n_blocks = 25
        args.n_feats = 64
        args.chop = True

    if args.template.find('MSRN') >= 0:
        args.model = 'MSRN'
        args.n_blocks = 8
        args.n_feats = 64
        args.chop = True
