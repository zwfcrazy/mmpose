dataset_info = dict(
    dataset_name='coco_xiaoice_school',
    paper_info=dict(
        author='',
        title='',
        container='',
        year='2023',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(
            name='left_big_toe',
            id=17,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        18:
        dict(
            name='left_small_toe',
            id=18,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        19:
        dict(
            name='left_heel',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        20:
        dict(
            name='right_big_toe',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        21:
        dict(
            name='right_small_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        22:
        dict(
            name='right_heel',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel'),
        23:
        dict(
            name='jaw_right',
            id=23,
            color=[255, 255, 255],
            type='upper',
            swap='jaw_left'),
        24:
        dict(
            name='jaw_middle',
            id=24,
            color=[255, 255, 255],
            type='upper',
            swap=''),
        25:
        dict(
            name='jaw_left',
            id=25,
            color=[255, 255, 255],
            type='upper',
            swap='jaw_right'),
        26:
        dict(
            name='lip_right',
            id=26,
            color=[255, 255, 255],
            type='upper',
            swap='lip_left'),
        27:
        dict(
            name='lip_left',
            id=27,
            color=[255, 255, 255],
            type='upper',
            swap='lip_right'),
        28:
        dict(
            name='left_thumb_root',
            id=28,
            color=[255, 128, 0],
            type='upper',
            swap='right_thumb_root'),
        29:
        dict(
            name='left_thumb_tip',
            id=29,
            color=[255, 128, 0],
            type='upper',
            swap='right_thumb_tip'),
        30:
        dict(
            name='left_forefinger_root',
            id=30,
            color=[255, 153, 255],
            type='upper',
            swap='right_forefinger_root'),
        31:
        dict(
            name='left_forefinger_tip',
            id=31,
            color=[255, 153, 255],
            type='upper',
            swap='right_forefinger_tip'),
        32:
        dict(
            name='left_pinky_finger_root',
            id=32,
            color=[0, 255, 0],
            type='upper',
            swap='right_pinky_finger_root'),
        33:
        dict(
            name='left_pinky_finger_tip',
            id=33,
            color=[0, 255, 0],
            type='upper',
            swap='right_pinky_finger_tip'),
        34:
        dict(
            name='right_thumb_root',
            id=34,
            color=[255, 128, 0],
            type='upper',
            swap='left_thumb_root'),
        35:
        dict(
            name='right_thumb_tip',
            id=35,
            color=[255, 128, 0],
            type='upper',
            swap='left_thumb_tip'),
        36:
        dict(
            name='right_forefinger_root',
            id=36,
            color=[255, 153, 255],
            type='upper',
            swap='left_forefinger_root'),
        37:
        dict(
            name='right_forefinger_tip',
            id=37,
            color=[255, 153, 255],
            type='upper',
            swap='left_forefinger_tip'),
        38:
        dict(
            name='right_pinky_finger_root',
            id=38,
            color=[0, 255, 0],
            type='upper',
            swap='left_pinky_finger_root'),
        39:
        dict(
            name='right_pinky_finger_tip',
            id=39,
            color=[0, 255, 0],
            type='upper',
            swap='left_pinky_finger_tip')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
        20:
        dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
        21:
        dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
        22:
        dict(
            link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
        23:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=23,
            color=[255, 128, 0]),
        24:
        dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
        25:
        dict(
            link=('left_wrist', 'left_thumb_root'), id=25, color=[255, 128,
                                                                  0]),
        26:
        dict(
            link=('left_thumb_root', 'left_thumb_tip'),
            id=26,
            color=[255, 128, 0]),
        27:
        dict(
            link=('left_wrist', 'left_forefinger_root'),
            id=27,
            color=[255, 153, 255]),
        28:
        dict(
            link=('left_forefinger_root', 'left_forefinger_tip'),
            id=28,
            color=[255, 153, 255]),
        29:
        dict(
            link=('left_wrist', 'left_pinky_finger_root'),
            id=29,
            color=[0, 255, 0]),
        30:
        dict(
            link=('left_pinky_finger_root', 'left_pinky_finger_tip'),
            id=30,
            color=[0, 255, 0]),
        31:
        dict(
            link=('right_wrist', 'right_thumb_root'),
            id=31,
            color=[255, 128, 0]),
        32:
        dict(
            link=('right_thumb_root', 'right_thumb_tip'),
            id=32,
            color=[255, 128, 0]),
        33:
        dict(
            link=('right_wrist', 'right_forefinger_root'),
            id=33,
            color=[255, 153, 255]),
        34:
        dict(
            link=('right_forefinger_root', 'right_forefinger_tip'),
            id=34,
            color=[255, 153, 255]),
        35:
        dict(
            link=('right_wrist', 'right_pinky_finger_root'),
            id=35,
            color=[0, 255, 0]),
        36:
        dict(
            link=('right_pinky_finger_root', 'right_pinky_finger_tip'),
            id=36,
            color=[0, 255, 0]),
    },
    joint_weights=[1.] * 40,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.068,
        0.066,
        0.066,
        0.092,
        0.094,
        0.094,
        0.025,
        0.020,
        0.023,
        0.008,
        0.01,
        0.035,
        0.047,
        0.026,
        0.035,
        0.02,
        0.031,
        0.035,
        0.047,
        0.026,
        0.035,
        0.02,
        0.031,
    ],
    pck_thr=[
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.2,
        0.2,
        0.15,
        0.15,
        0.1,
        0.1,
        0.4,
        0.4,
        0.2,
        0.2,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ])
