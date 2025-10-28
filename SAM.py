## 安装SAM的两种方法 
# 1. 
# pip install git+https://github.com/facebookresearch/segment-anything.git
# 2. 先后运行下面这两行指令
# git clone git@github.com:facebookresearch/segment-anything.git
# cd segment-anything; pip install -e .
'''上面来源SAM的github readme, 要是下载不成功可以去原github上看看'''

'''还要安装下面这一行的库'''
#pip install opencv-python pycocotools matplotlib


from segment_anything import SamPredictor, sam_model_registry
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
from tqdm import *
print('Successfully import all requirements!')


join = os.path.join
device = 'cuda'

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.savefig('test.png')
    ax.imshow(mask_image)
    
def show_points(coords, labels,marker_size=375):
    pos_points = coords[labels==1]
    print('pos',pos_points)
    neg_points = coords[labels==0]
    print('neg',neg_points)
    plt.figure(figsize=(2.24, 2.24))
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    plt.savefig('fuk.png')
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 



def mask_pics(input_path,output_path): 
    for j in range(len(input_path)):   
        path_real = input_path[j] + '/0_real'
        path_fake = input_path[j] + '/1_fake'
        path_l = [path_real,path_fake]
        path_o = [output_path[j] + '/0_real',output_path[j] + '/1_fake'] 
        for k in range(2):  
            file_list = sorted(os.listdir(path_l[k]))
            for i in tqdm(range(len(file_list))): 
                full_p = join(path_l[k],file_list[i])
                sam = sam_model_registry["default"](checkpoint="/mntnfs/med_data5/wanyue/sam_vit_h_4b8939.pth")
                sam.to(device=device)
                predictor = SamPredictor(sam)
                image = cv2.imread(full_p)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                input_point = np.array([[128,128],[128,236]])
                input_label = np.array([1,1])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label, 
                    multimask_output=True,
                )

                # choose best score
                mask_input = logits[np.argmax(scores), :, :] # choose best mask

                masks, _, _ = predictor.predict( 
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )  # masks
                masks = np.squeeze(masks.astype(int)) # best mask

                # mask on pics
                h,w = masks.shape
                img2 = np.zeros_like(image)
                img2[:,:,0] = masks 
                img2[:,:,1] = masks
                img2[:,:,2] = masks
                img_fuck = img2 * image
                img_fk = cv2.cvtColor(img_fuck,cv2.COLOR_RGB2BGR)
                cv2.imwrite(join(path_o[k],file_list[i]),img_fk)




input_path=['./real-vs-fake/train','./real-vs-fake/valid','./real-vs-fake/test']  
# 需要先创建输出路径./results_mask/train/0_real,./results_mask/train/1_fake
output_path=['./real-vs-fake/results_mask/train','./real-vs-fake/results_mask/val','./real-vs-fake/results_mask/test'] 

mask_pics(input_path,output_path)

