dataset_info = dict(
    dataset_name='cow16',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2024',
        homepage='httpxxx'
    ),
    keypoint_info={
        0:
        dict(name='Head_top', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='Neck',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='Spine',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='Right_front_leg_root',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='Left_front_leg_root'),
        4:
        dict(
            name='Right_front_knee',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap='Left_front_knee'),
        5:
        dict(
            name='Right_front_hoof',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='Left_front_hoof'),
        6:
        dict(
            name='Left_front_leg_root',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='Right_front_leg_root'),
        7:
        dict(
            name='Left_front_knee',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='Right_front_knee'),
        8:
        dict(
            name='Left_front_hoof',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='Right_front_hoof'),
        9:
        dict(
            name='Coccyx',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        10:
        dict(
            name='Right_hind_leg_root',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='Left_hind_leg_root'),
        11:
        dict(
            name='Right_hind_knee',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='Left_hind_knee'),
        12:
        dict(
            name='Right_hind_hoof',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='Left_hind_hoof'),
        13:
        dict(
            name='Left_hind_leg_root',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='Right_hind_leg_root'),
        14:
        dict(
            name='Left_hind_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='Right_hind_knee'),
        15:
        dict(
            name='Left_hind_hoof',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='Right_hind_hoof'),
    },
    skeleton_info={
        0:
        dict(link=('Head_top', 'Neck'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('Neck', 'Spine'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('Spine', 'Right_front_leg_root'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('Spine', 'Left_front_leg_root'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('Spine', 'Coccyx'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('Right_front_leg_root', 'Right_front_knee'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('Right_front_knee', 'Right_front_hoof'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('Left_front_leg_root', 'Left_front_knee'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('Left_front_knee', 'Left_front_hoof'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('Right_hind_leg_root', 'Right_hind_knee'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('Right_hind_knee', 'Right_hind_hoof'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('Left_hind_leg_root', 'Left_hind_knee'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('Left_hind_knee', 'Left_hind_hoof'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('Coccyx', 'Right_hind_leg_root'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('Coccyx', 'Left_hind_leg_root'), id=14, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087
    ])
