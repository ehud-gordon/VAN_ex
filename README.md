## GraphSLAM
Performs SLAM on a set of stereo images, finds cameras locations using a combination of PnP, Bundle Adjustment,
pose graph and loop closures. Detailed explanation in the pdf under /docs.

## Usage
Dependencies are listed in `reuquirements.txt`
Then run:  
```python main.py [--dataset_path path] [--out_dir path]```  
`dataset_path` should point to a directory holding files
in kitti's format. 

## Abbreviations
| Abbreviation      | Meaning    |  
|:-----------------:|:------------------:|
i, j | two consecutive frames/cameras, i comes before j
cond | conditional
cov | covariance
CS | coordinate system
sf | stereo features (keypoints + point-cloud)
pc | point cloud
k | the intrinsic camera matrix
ext | extrinsics matrix
(m,n) | ndarray of shape (m,n)
dw | location of camera's origin w.r.t world coordinates
mat | matrix
rot | rotation
trans | translation









