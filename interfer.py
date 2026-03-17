from mmpose.apis import MMPoseInferencer


#  URL 
inferencer = MMPoseInferencer(
    pose2d='/data/configs/cid_w32.py',
    pose2d_weights='/data/work_dirs/cid_w32/best_coco_AP_epoch_130.pth'
)
folder_path ='./video/18.mp4'
result_generator = inferencer(folder_path, show=False,vis_out_dir='./vis_reults')
result = [result for result in result_generator]