import albumentations as A
from albumentations.pytorch.transforms import ToTensor

def get_train_transform(img_size):

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

    return train_transform

def get_valid_transform(img_size):

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
