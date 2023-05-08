_base_ = [
    '../_base_/models/mocov3_vs_generator.py',
    '../_base_/datasets/imagenet_mocov3_vs_gen.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(dataset=dict(type='mmselfsup.ImageList', ann_file='filenames.txt'), batch_size=24, num_workers=8)

#data_preprocessor =

# optimizer
optimizer = dict(type='AdamW', lr=2.4e-4, weight_decay=0.1)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'bn.': dict(decay_mult=0, lars_exclude=True),
            'project.1': dict(decay_mult=0, lars_exclude=True),
            'bn1': dict(decay_mult=0, lars_exclude=True),
            'bn2': dict(decay_mult=0, lars_exclude=True),
        }),
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=98,
        by_epoch=True,
        begin=2,
        end=100,
        convert_to_iter_based=True)
]
# Initialize according to GAN training
model = dict(backbone=dict(init_cfg=dict(type='Normal', std=0.02)))

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

# runtime settings
# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=5))

vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
