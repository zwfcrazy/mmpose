_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/coco_xiaoice_school.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='all/AP', rule='greater'),
    visualization=dict(type='PoseVisualizationHook', enable=True, enable_train=True, num_images=3),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/user/zhuwenfei/human_pose/weights/hrnet_w32_coco_wholebody_256x192-853765cd_20200918_dev-1.x.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=40,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
        output_heatmaps=True
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/mnt/user/zhuwenfei/human_pose/data/school/pose/cocktail/'

# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args={{_base_.file_client_args}}),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(
        type='Albumentation',
        transforms=[
            dict(
                type='CoarseDropout',
                max_holes=8,
                max_height=40,
                max_width=40,
                min_holes=1,
                min_height=10,
                min_width=10,
                p=0.5),
        ]),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs', pack_transformed=True)
]
val_pipeline = [
    dict(type='LoadImage', file_client_args={{_base_.file_client_args}}),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=_base_.dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        metainfo=_base_.dataset_info,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val_coco.json',
        data_prefix=dict(img='val_coco/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'val.json')

val_evaluator = dict(
    type='PartitionMetric',
    metric=dict(
        type='CocoMetric',
    ),
    partitions=dict(
        body=range(17),
        foot=range(17, 23),
        jaw_lip=range(23, 28),
        left_hand=range(28, 34),
        right_hand=range(34, 40),
        all=range(40)
    )
)
test_evaluator = val_evaluator
