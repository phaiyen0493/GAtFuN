To create synthetic occlusions on Human3.6M, you will need to get access to the full dataset and extract each video to frames:
```bash
python vid_extract.py
```

To create occlusion object to each frame, please run the notebook `create_synthetic_occlusion_to_Human3_6m_frames.ipynb`. You can change the radius_level and time_level to `1,2,3`, accordingly to the Size and Duration of the synthetic object.

`Detectron2_2D_Detection.ipynb` is used to extract 2D keypoints as inputs to test our model's adaptability under synthetic occlusions, and on in-the-wild videos. Make sure you install Detectron2:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For visualization, check out `draw_2D_skeletons.ipynb` and `draw_3D_skeleton.ipynb`.
