# UniMVSNet
UniMVSNet is a learning-based multi-view stereo model, which has a unified depth representation to not only achieve sub-pixel depth estimation but also constrain the cost volume directly.
To excavate the potential of our novel representation, we designed a Unified Focal Loss to combat the challenge of sample imbalance more reasonably and uniformly.
Details are described in our paper:
> Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation and Focal Loss
>
> Rui Peng, Rongjie Wang, Zhenyu Wang, Yawen Lai, Ronggang Wang
>
> CVPR 2022 ([arxiv](https://arxiv.org/abs/2201.01501))

<p align="center">
    <img src="./.github/images/sample.png" width="100%"/>
</p>

UniMVSNet is more robust on the challenge regions and can generate more 
accurate depth maps. The point cloud is more complete and the details are finer.

*If there are any errors in our code, please feel free to ask your questions.*

## ⚙ Setup
#### 1. Recommended environment
- PyTorch 1.2
- Python 3.6

#### 2. DTU Dataset

**Training Data**. We adopt the full resolution ground-truth depth provided in CasMVSNet or MVSNet. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). 
Unzip them and put the `Depth_raw` to `dtu_training` folder. The structure is just like:
```
dtu_training                          
       ├── Cameras                
       ├── Depths   
       ├── Depths_raw
       └── Rectified
```
**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it. The structure is just like:
```
dtu_testing                          
       ├── Cameras                
       ├── scan1   
       ├── scan2
       ├── ...
```

#### 3. BlendedMVS Dataset

**Training Data** and **Validation Data**. Download [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) and 
unzip it. And we only adopt 
BlendedMVS for finetuning and not testing on it. The structure is just like:
```
blendedmvs                          
       ├── 5a0271884e62597cdee0d0eb                
       ├── 5a3ca9cb270f0e3f14d0eddb   
       ├── ...
       ├── training_list.txt
       ├── ...
```

#### 4. Tanks and Temples Dataset

**Testing Data**. Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and 
unzip it. Here, we adopt the camera parameters of short depth range version (Included in your download), therefore, you should 
replace the `cams` folder in `intermediate` folder with the short depth range version manually. The 
structure is just like:
```
tanksandtemples                          
       ├── advanced                 
       │   ├── Auditorium       
       │   ├── ...  
       └── intermediate
           ├── Family       
           ├── ...          
```

## 📊 Testing

#### 1. Download models
Download our pretrained model and put it to `<your model path>`.
<table align="center">
  	<tr align="center">
  	    <td>Pre-trained</td>
  	    <td>Train Ndepth</td>
		<td>Train View</td>
		<td>Test Ndepth</td>
		<td>Test View</td>
		<td colspan="2">Performance</td>
	</tr>
	<tr align="center">
	    <td><a href="https://drive.google.com/file/d/1DdJa4ZeS_E3BfAYpozAHc3qXnDqrkLFZ/view?usp=sharing">unimvsnet_dtu</a></td>
		<td>48-32-8</td>
		<td>5</td>
		<td>48-32-8</td>
		<td>5</td>
		<td>DTU↓</td>
		<td>0.315</td>
	</tr>
	<tr align="center">
	    <td rowspan="2"><a href="https://drive.google.com/file/d/1j0iahbAYNLS1871kAwUDrB4YTCMC6N3d/view?usp=sharing">unimvsnet_blendedmvs</a></td>
		<td rowspan="2">48-32-8</td>
		<td rowspan="2">7</td>
		<td rowspan="2">64-32-8</td>
		<td rowspan="2">11</td>
		<td>Intermediate↑</td>
		<td>64.36</td>
	</tr>
	<tr align="center">
	    <td>Advanced↑</td>
		<td>38.96</td>
	</tr>
</table>

**Note:** `unimvsnet_blendedmvs` is the model trained on DTU and then finetuned 
on BlendedMVS, and we only used it to test on Tanks dataset. More details 
can be found in our paper.

#### 2. DTU testing

**Point generation**. To recreate the results from our paper, you need to specify the `datapath` to 
`<your dtu_testing path>`, `outdir` to `<your output save path>` and `resume` 
 to `<your model path>` in shell file `./script/dtu_test.sh` first and then run:
```
bash ./scripts/dtu_test.sh
```

We adopt Gipuma to filter and fuse DTU point clouds, and more details can be found in 
our code. Note that we use the unimvsnet_dtu checkpoint when testing on DTU.

**Point testing**. You need to move the point clouds generated under each scene into a 
folder `dtu_points`. Meanwhile, you need to rename the point cloud in 
the **mvsnet001_l3.ply** format (the middle three digits represent the number of scene).
Then specify the `dataPath`, `plyPath` and `resultsPath` in 
`./dtu_eval/BaseEvalMain_web.m` and `./dtu_eval/ComputeStat_web.m`. Finally, run 
file `./dtu_eval/BaseEvalMain_web.m` through matlab software to evaluate 
DTU point scene by scene first, then execute file `./dtu_eval/BaseEvalMain_web.m` 
to get the average metrics for the entire dataset.

#### 3. Tanks and Temples testing

**Point generation**. Similarly, you need specify the `datapath`, `outdir` and `resume` in shell file 
`./scripts/tank_test.sh`, and then run:
```
bash ./scripts/tank_test.sh
```
Note that we use the unimvsnet_blendedmvs checkpoint when testing on Tanks and Temples.

**Point testing**. You need to upload the generated points to Tanks and Temples 
benchmark, and it will return test results within a few hours.

We adopt dynamic geometric consistency checking strategies to filter and 
fuse Tanks point clouds. Meanwhile, we consider the photometric constrains 
of all stages like VisMVSNet. The configuration of scenes in `./fiter/tank_test_config.py` 
is the closest we get to reproducing our baseline. 

<table align="center">
  	<tr align="center">
  	    <td>Model</td>
  	    <td>Test View</td>
  	    <td>Mean↑</td>
		<td>Fam.↑</td>
		<td>Fra.↑</td>
		<td>Hor.↑</td>
		<td>Lig.↑</td>
		<td>M60↑</td>
		<td>Pan.↑</td>
		<td>Pla.↑</td>
		<td>Train↑</td>
	</tr>
	<tr align="center">
	    <td rowspan="2">CasMVSNet</td>
	    <td>7</td>
		<td>56.41</td>
		<td>76.58</td>
		<td>60.14</td>
		<td>47.26</td>
		<td>57.94</td>
		<td>56.68</td>
		<td>51.23</td>
		<td>52.28</td>
		<td>49.17</td>
	</tr>
	<tr align="center">
	    <td>11</td>
		<td>55.96</td>
		<td>75.44</td>
		<td>59.57</td>
		<td>43.45</td>
		<td>59.07</td>
		<td>57.03</td>
		<td>52.23</td>
		<td>52.15</td>
		<td>48.71</td>
	</tr>
</table>

## 📦 DTU points
You can download our precomputed DTU point clouds from the following link:

<table align="center">
  	<tr align="center">
  	    <td>Points</td>
  	    <td>Confidence Threshold</td>
  	    <td>Consistent View</td>
  	    <td>Accuracy↓</td>
  	    <td>Completeness↓</td>
		<td>Overall↓</td>
	</tr>
	<tr align="center">
	    <td><a href="https://drive.google.com/drive/folders/1kK0qcT4d1Qm5Uc25IxgDlwfPTm_Z-sv3?usp=sharing">dtu_points</a></td>
		<td>0.3</td>
		<td>3</td>
		<td>0.352</td>
		<td>0.278</td>
		<td>0.315</td>
	</tr>
</table>

## 🖼 Visualization

To visualize the depth map in pfm format, run:
```
python main.py --vis --depth_path <your depth path> --depth_img_save_dir <your depth image save directory>
```
The visualized depth map will be saved as `<your depth image save directory>/depth.png`. For visualization of point clouds, 
some existing software such as MeshLab can be used.

## ⏳ Training

#### 1. DTU training

To train the model from scratch on DTU, specify the `datapath` and `log_dir` 
in `./scripts/dtu_train.sh` first 
and then run:
```
bash ./scripts/dtu_train.sh
```
By default, we employ the *DistributedDataParallel* mode to train our model, you can also 
train your model in a single GPU.

#### 2. BlendedMVS fine-tuning

To fine-tune the model on BlendedMVS, you need specify `datapath`, `log_dir` and
`resume` in `./scripts/blendedmvs_finetune.sh` first, then run:
```
bash ./scripts/blendedmvs_finetune.sh
```
Actually, you can train the model on BlendedMVS from scratch just like some 
other methods through removing the command `resume`.


## ⚖ Citation
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{unimvsnet,
    title = {Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation and Focal Loss},
    author = {Peng, Rui and Wang, Rongjie and Wang, Zhenyu and Lai, Yawen and Wang, Ronggang},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```

## 👩‍ Acknowledgements

Thanks to [MVSNet](https://github.com/YoYo000/MVSNet), [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch) and [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet).

