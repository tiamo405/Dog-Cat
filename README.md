# 1. Data
```
data __train  __Dog   __img1.jpg
             |       |__img2.jpg
             |       |.....
             |__Cat  __img1.jpg    
                     |__img2.jpg    
                     |.....
__test  __img1.jpg
        |.....
```
- dowload data
```
bash dowload_data.sh
```
# 2. Train
```
python train.py --data_root data/train --name_model resnet101 --checkpoint_dir checkpoints
```
- folder weight :
```
checkpoint_dir __ name_model __ times train __ number epoch.pth
ex : checkpoints/resnet101/0001/1.pth
```
# 3. Test

```
python predict.py --checkpoint_dir checkpoints --name_model resnet101 --num_train 0001 --num_ckpt 1 --image data/test/tai-anh-cho-dep.webp
``` 
* Test on colab
```
--checkpoint_dir : drive/weight/DandC
```
## Trên drive lưu file weight như sau:
```
drive/weight/DandC/resnet101/0001/1.pth
```
[Dowload pretrained](https://drive.google.com/drive/folders/1kSdUx6yzp7eEaAJrdeA_gsLfgCLT5uWq?usp=sharing)
# Colab
[Colab demo](https://colab.research.google.com/drive/1bSoSouDuzIPWZ713MrssSIkX_VepZBwW?usp=sharing)