import os
from torchvision import transforms
from transforms import *
from casme3 import casme3_VideoClsDataset
from affect_net import affect_net_VideoClsDataset
import my_utils
from other_dataset import other_VideoClsDataset

class DataAugmentationForVideo_(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideo_,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideo_(args)
    dataset = Video_(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
        
        
        
    elif args.data_set == 'affectnet':
        mode = None
        anno_path = None
        if args.nb_classes==7:
            if is_train is True:
                mode = 'train'
                anno_path = os.path.join("YOUR_PATH/7_classes_fix_miss.csv")
            elif test_mode is True:
                mode = 'test'
                anno_path = anno_path = os.path.join("YOUR_PATH/7_classes_val_set.csv")
            else:  
                mode = 'validation'
                anno_path = anno_path = os.path.join("YOUR_PATH/7_classes_val_set.csv")

        dataset = affect_net_VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes           
    elif args.data_set == 'smic':
        mode = None
        anno_path = None

        print("---------------------going to loso on smic-----------------------")
        if args.nb_classes==3:
            if is_train is True:
                mode = 'train'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'train_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/smic_label"
            elif test_mode is True:
                mode = 'test'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/smic_label"
            else:  
                mode = 'validation'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/smic_label"

            dataset =other_VideoClsDataset(
                anno_path=anno_path,
                data_path='/',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
            nb_classes = args.nb_classes       
           
    elif args.data_set == 'samm':
        mode = None
        anno_path = None

        print("---------------------going to loso on smic-----------------------")
        if args.nb_classes==3:
            if is_train is True:
                mode = 'train'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'train_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/samm_label"
            elif test_mode is True:
                mode = 'test'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/samm_label"
            else:  
                mode = 'validation'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/samm_label"

            dataset =other_VideoClsDataset(
                anno_path=anno_path,
                data_path='/',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
            nb_classes = args.nb_classes     
            
    elif args.data_set == 'casme2':
        mode = None
        anno_path = None

        print("---------------------going to loso on smic-----------------------")
        if args.nb_classes==3:
            if is_train is True:
                mode = 'train'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'train_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/casme2_label"
            elif test_mode is True:
                mode = 'test'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/casme2_label"
            else:  
                mode = 'validation'
                # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                anno_path="YOUR_PATH/casme2_label"

            dataset =other_VideoClsDataset(
                anno_path=anno_path,
                data_path='/',
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
            nb_classes = args.nb_classes  
                    
    elif args.data_set == 'casme3':
        mode = None
        anno_path = None
        if args.k_folds!=0:
            if args.nb_classes==7:
                if is_train is True:
                    mode = 'train'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'train_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)
                elif test_mode is True:
                    mode = 'test'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)
                else:  
                    mode = 'validation'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)

                dataset = casme3_VideoClsDataset(
                    anno_path=anno_path,
                    data_path='/',
                    mode=mode,
                    clip_len=args.num_frames,
                    frame_sample_rate=args.sampling_rate,
                    num_segment=1,
                    test_num_segment=args.test_num_segment,
                    test_num_crop=args.test_num_crop,
                    num_crop=1 if not test_mode else 3,
                    keep_aspect_ratio=True,
                    crop_size=args.input_size,
                    short_side_size=args.short_side_size,
                    new_height=256,
                    new_width=320,
                    args=args)
                nb_classes = args.nb_classes 
            elif args.nb_classes==3:
                if is_train is True:
                    mode = 'train'
                    anno_path = os.path.join("YOUR_PATH/wash_casme3/4fold_cross",f'balanced_train_set_{args.k_counts+1}.csv')
                elif test_mode is True:
                    mode = 'test'
                    anno_path = os.path.join("YOUR_PATH/wash_casme3/4fold_cross",f'test_set_{args.k_counts+1}.csv')
                else:  
                    mode = 'validation'
                    anno_path = os.path.join("YOUR_PATH/wash_casme3/4fold_cross",f'test_set_{args.k_counts+1}.csv')

                dataset = casme3_VideoClsDataset(
                    anno_path=anno_path,
                    data_path='/',
                    mode=mode,
                    clip_len=args.num_frames,
                    frame_sample_rate=args.sampling_rate,
                    num_segment=1,
                    test_num_segment=args.test_num_segment,
                    test_num_crop=args.test_num_crop,
                    num_crop=1 if not test_mode else 3,
                    keep_aspect_ratio=True,
                    crop_size=args.input_size,
                    short_side_size=args.short_side_size,
                    new_height=256,
                    new_width=320,
                    args=args)
                nb_classes = args.nb_classes                
                #During non cross validation
        elif args.k_folds==0:
            print("---------------------going to loso-----------------------")
            if args.nb_classes==7:
                if is_train is True:
                    mode = 'train'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'train_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)
                elif test_mode is True:
                    mode = 'test'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)
                else:  
                    mode = 'validation'
                    # anno_path = os.path.join("YOUR_PATH/wash_casme3/5fold_7classes",f'test_set_{args.k_counts+1}.csv')
                    anno_path=my_utils.get_casme3_label(args,mode)

                dataset = casme3_VideoClsDataset(
                    anno_path=anno_path,
                    data_path='/',
                    mode=mode,
                    clip_len=args.num_frames,
                    frame_sample_rate=args.sampling_rate,
                    num_segment=1,
                    test_num_segment=args.test_num_segment,
                    test_num_crop=args.test_num_crop,
                    num_crop=1 if not test_mode else 3,
                    keep_aspect_ratio=True,
                    crop_size=args.input_size,
                    short_side_size=args.short_side_size,
                    new_height=256,
                    new_width=320,
                    args=args)
                nb_classes = args.nb_classes 
        
        
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
