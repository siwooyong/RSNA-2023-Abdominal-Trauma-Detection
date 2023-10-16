class CustomAug(nn.Module):
    def __init__(self, prob = 0.5, s = 224):
        super(CustomAug, self).__init__()
        self.prob = prob

        self.do_random_rotate = v2.RandomRotation(
            degrees = (-45, 45),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        self.do_random_scale = v2.ScaleJitter(
            target_size = [s, s],
            scale_range = (0.8, 1.2),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            antialias = True)

        self.do_random_crop = v2.RandomCrop(
            size = [s, s],
            #padding = None,
            pad_if_needed = True,
            fill = 0,
            padding_mode = 'constant'
        )

        self.do_horizontal_flip = v2.RandomHorizontalFlip(self.prob)
        self.do_vertical_flip = v2.RandomVerticalFlip(self.prob)
    def forward(self, x):
        if np.random.rand() < self.prob:
            x = self.do_random_rotate(x)

        if np.random.rand() < self.prob:
            x = self.do_random_scale(x)
            x = self.do_random_crop(x)

        x = self.do_horizontal_flip(x)
        x = self.do_vertical_flip(x)
        return x

aug_function = CustomAug()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=False):
        self.df = df
        self.label_columns = self.df.columns[1:-2]

        #self.down_sampling = 1
        self.max_frame = 256
        self.img_size = 224

        self.augmentation = augmentation

        self.sample_weights = {
            'bowel' : {0:1, 1:2},
            'extravasation' : {0:1, 1:6},
            'kidney' : {0:1, 1:2, 2:4},
            'liver' : {0:1, 1:2, 2:4},
            'spleen' : {0:1, 1:2, 2:4},
            'any_injury' : {0:1, 1:6}
            }

        self.sample_weights = {
            'bowel' : {0:1, 1:1},
            'extravasation' : {0:1, 1:1},
            'kidney' : {0:1, 1:1, 2:1},
            'liver' : {0:1, 1:1, 2:1},
            'spleen' : {0:1, 1:1, 2:1},
            'any_injury' : {0:1, 1:1}
            }

    def __len__(self):
        return len(self.df)

    def get_stride_box(self, min_y, min_x, max_y, max_x, stride=10):
        min_y = np.clip(min_y - stride, a_min=0, a_max=512)
        min_x = np.clip(min_x - stride, a_min=0, a_max=512)
        max_y = np.clip(max_y + stride, a_min=0, a_max=512)
        max_x = np.clip(max_x + stride, a_min=0, a_max=512)
        return min_y, min_x, max_y, max_x

    def get_cropped_organs(self, video, box, ratio=(512/320)):
        organs = []
        for i in range(box.shape[0]):
          min_z, min_y, min_x, max_z, max_y, max_x = box[i]
          if 0.0 not in [max_z - min_z, max_y - min_y, max_x - min_x]:
            min_y, min_x, max_y, max_x = int(ratio*min_y), int(ratio*min_x), int(ratio*max_y), int(ratio*max_x)
            min_y, min_x, max_y, max_x = self.get_stride_box(min_y, min_x, max_y, max_x)
            print(max_z-min_z, max_y-min_y, max_x-min_x)
            organ = video[min_z:max_z, min_y:max_y, min_x:max_x]
          else:
            organ = video

          organ = F.interpolate(
              organ.unsqueeze(0).unsqueeze(0),
              size=[96, 224, 224],
              mode='trilinear'
              ).squeeze(0).squeeze(0)
          organs.append(organ)
        return organs

    def load_image(self, image_path, img_size=512):
        return cv2.resize(cv2.imread(image_path)[:,:,0], dsize=(img_size, img_size))


    def __getitem__(self, index):
        sample = self.df.loc[index]
        patient_id, series_id, any_injury = int(sample['patient_id']), int(sample['series_id']), int(sample['any_injury'])

        images = torch.tensor(np.load(f'/content/train_videos/{patient_id}/{series_id}_images.npy', mmap_mode='r'), dtype=torch.float)
        crop_liver = torch.tensor(np.load(f'/content/train_videos/{patient_id}/{series_id}_liver.npy', mmap_mode='r'), dtype=torch.float)
        crop_spleen = torch.tensor(np.load(f'/content/train_videos/{patient_id}/{series_id}_spleen.npy', mmap_mode='r'), dtype=torch.float)
        crop_kidney = torch.tensor(np.load(f'/content/train_videos/{patient_id}/{series_id}_kidney.npy', mmap_mode='r'), dtype=torch.float)

        if self.augmentation:
          images = aug_function(images)
          crop_liver = aug_function(crop_liver)
          crop_spleen = aug_function(crop_spleen)
          crop_kidney = aug_function(crop_kidney)

        label = torch.tensor(sample[self.label_columns].values, dtype=torch.long)

        bowel = label[0:2].argmax()
        extravasation = label[2:4].argmax()
        kidney = label[4:7].argmax()
        liver = label[7:10].argmax()
        spleen = label[10:13].argmax()
        any_injury = torch.tensor(any_injury, dtype=torch.float)

        sample_weights = torch.tensor([
            self.sample_weights['bowel'][bowel.tolist()],
            self.sample_weights['extravasation'][extravasation.tolist()],
            self.sample_weights['kidney'][kidney.tolist()],
            self.sample_weights['liver'][liver.tolist()],
            self.sample_weights['spleen'][spleen.tolist()],
            self.sample_weights['any_injury'][any_injury.tolist()]
        ])

        images, crop_liver, crop_spleen, crop_kidney = images/255.0, crop_liver/255.0, crop_spleen/255.0, crop_kidney/255.0

        return images, crop_liver, crop_spleen, crop_kidney, label, bowel, extravasation, kidney, liver, spleen, any_injury, sample_weights
