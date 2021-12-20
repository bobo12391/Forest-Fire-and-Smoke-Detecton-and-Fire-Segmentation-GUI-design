# Train your own unet model

## Preparation

### My datasets(a little small, so the accuracy is not great):
1. Find the "train.zip" in the U-Net/data/train
2. Extract "train.zip" in U-Net/data/train. You can see that the folder structure is like "U-Net/data/train/000.jpg".
3. Then delete the "readme.md" and "train.zip"
4. Process the followings:

```markdown
cd U-Net
python main.py --train 1
```
### Train your own datasets
Just keep the datasets structure as following like:
```markdown
Original images: U-Net/data/train/000.jpg
  
Mask images: U-Net/data/train/000_mask.jpg
```
For example:
 
If you have 101 images to train, the trainning images need to sort in order from "000" to "100". 

Process the followings:
```markdown
cd U-Net
python main.py --train 1
```
