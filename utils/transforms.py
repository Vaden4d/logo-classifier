import albumentations as A
from albumentations.pytorch.transforms import ToTensor

def get_train_transform(img_size):

    train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensor()])

    return train_transform

def get_valid_transform(img_size):
    # constants from imagenet data
    test_transform = A.Compose([A.Resize(img_size, img_size),
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]
                                        ),
                                    ToTensor()
                                    ])

    return test_transform

def get_transforms(img_size):

    train_transform = A.Compose([A.Resize(img_size, img_size),
                                    A.HorizontalFlip(),
                                    A.VerticalFlip(),
                                    A.ShiftScaleRotate(),
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]
                                        ),
                                    ToTensor()
                                    ])
    test_transform = A.Compose([A.Resize(img_size, img_size),
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]
                                        ),
                                    ToTensor()
                                    ])

    return train_transform, test_transform
