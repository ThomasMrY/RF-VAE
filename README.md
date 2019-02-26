# RF VAE
Pytorch implementation of RF_VAE proposed in Relevance Factor VAE: Learning and Identifying Disentangled Factors, Kim et al.([https://arxiv.org/abs/1902.01568])
<br>

### Dependencies
```
python 3.6.4
pytorch 0.4.0 (or check pytorch-0.3.1 branch for pytorch 0.3.1)
visdom
tqdm
```
<br>

### Datasets
1. 2D Shapes(dsprites) Dataset
```
sh scripts/prepare_data.sh dsprites
```
2. 3D Chairs Dataset
```
sh scripts/prepare_data.sh 3DChairs
```
3. CelebA Dataset([download])
```
# first download img_align_celeba.zip and put in data directory like below
└── data
    └── img_align_celeba.zip

# then run scrip file
sh scripts/prepare_data.sh CelebA
```

then data directory structure will be like below<br>
```
.
└── data
    └── dsprites-dataset
        └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
    ├── 3DChairs
        └── images
            ├── 1_xxx.png
            ├── 2_xxx.png
            ├── ...
    ├── CelebA
        └── img_align_celeba
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 202599.jpg
    └── ...
```
NOTE: I recommend to preprocess image files(e.g. resizing) BEFORE training and avoid preprocessing on-the-fly.
<br>

### Usage
initialize visdom
```
python -m visdom.server
```
you can reproduce results below as follows
```
e.g.
sh scripts/run_celeba.sh $RUN_NAME
sh scripts/run_dsprites_gamma6p4.sh $RUN_NAME
sh scripts/run_dsprites_gamma10.sh $RUN_NAME
sh scripts/run_3dchairs.sh $RUN_NAME
```
or you can run your own experiments by setting parameters manually
```
e.g.
python main.py --name run_celeba --dataset celeba --gamma 6.4 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 ...
```
check training process on the visdom server
```
localhost:8097
```
<br>

##### visdom line plot
![dsprites_plot](result/Capture.png)

![dsprites_plot](result/distribute.png)
##### latent traversal gif(```--save_output True```)
<p align="center">
<img src=result/random_img.gif>
<img src=result/fixed_heart.gif>
<img src=result/fixed_square.gif>
<img src=result/fixed_ellipse.gif>
</p>


##### reconstruction(left: true, right: reconstruction)
<p align="center">
<img src=result/300000.gif>
</p>

### Reference
1. Relevance Factor VAE: Learning and Identifying Disentangled Factors.([https://arxiv.org/abs/1902.01568])


[https://arxiv.org/abs/1902.01568]: https://arxiv.org/abs/1902.01568
[download]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
