import os
import random
import matplotlib.pyplot as plt

def load_images(path1, path2, num_images=32):
    images1 = []
    images2 = []
    for i in range(num_images):
        img_path1 = os.path.join(path1, f'batch_0_img_{i}.png')
        img_path2 = os.path.join(path2, f'batch_0_img_{i}.png')
        if os.path.exists(img_path1) and os.path.exists(img_path2):
            img1 = plt.imread(img_path1)
            img2 = plt.imread(img_path2)
            images1.append(img1)
            images2.append(img2)
    return images1, images2

def plot_images(input_images, gt_images, model1_images, model2_images, num_samples=8):
    indices = random.sample(range(len(input_images)), num_samples)
    
    # 使用 constrained_layout 替代 tight_layout
    fig, axes = plt.subplots(
        num_samples, 4, 
        figsize=(12, 2.5 * num_samples), 
        constrained_layout=True  # 启用 constrained_layout
    )
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(input_images[idx])
        axes[i, 0].set_title('Input Image', fontsize=8)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_images[idx], cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=8)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(model1_images[idx], cmap='gray')
        axes[i, 2].set_title('Model1 Result', fontsize=8)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(model2_images[idx], cmap='gray')
        axes[i, 3].set_title('Model2 Result', fontsize=8)
        axes[i, 3].axis('off')
    
    # plt.show()
    plt.savefig("./examples/0out_model_compare.png")

# 路径
path1 = '/data/hongboye/projects/hook_poly_1_CMUNeXt_rwkv_1_3_128_256_768_no_shift'
path2 = '/data/hongboye/projects/hook_poly_1_CMUNeXt_rwkv_1_3_128_256_768'

# 加载图片
input_images, gt_images = load_images(path1, path2)
model1_images, model2_images = load_images(path1, path2)  # 假设model1和model2的结果也在同样的路径下

# 画出图片
plot_images(input_images, gt_images, model1_images, model2_images)