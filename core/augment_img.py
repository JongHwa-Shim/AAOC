from augmentation import *
import torchvision as tv
import PIL.Image as Image

src_dir = r'D:\Onedrive\OneDrive - 고려대학교\MiLab\research\내 논문\data augmentation\figure\grid images\stochastic'
trg_dir = r'D:\Onedrive\OneDrive - 고려대학교\MiLab\research\내 논문\data augmentation\figure\grid images\stochastic_augmented'

if __name__ == '__main__':
    # for single augmentation
    aug_ins = aug()
    aug_func = aug_ins.augmentation
    # for stocastic overlapping augmentation
    #aug_func = over_aug
    img_t_list = []
    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        img_pil = Image.open(img_path)
        img_pil = img_pil.convert('RGB')
        img_t = tv.transforms.ToTensor()(img_pil).unsqueeze(dim=0)

        aug_img_t, _ = aug_func(img_t)
        aug_img_t = aug_img_t.squeeze(dim=0)

        aug_img_pil = tv.transforms.ToPILImage()(aug_img_t)
        aug_img_pil.save(os.path.join(trg_dir, img_name))
