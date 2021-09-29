import sys
import os
import datetime
import argparse
import json

from dataset import Dataset, EarthBG, AugmentedDataset



def parse_args():

    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', required=True, help='train config file path')
    parser.add_argument('--src-dir', required= True, help='the dir to original images')
    parser.add_argument('--dst-dir', required= True, help='the destination dir to save corrupted images')

    args = parser.parse_args()

    is_dir([args.src_dir])
    is_file([args.config])
    os.makedirs(args.dst_dir, exist_ok=True)
    
    assert (os.path.splitext(args.config)[-1]=='.json'), f"{args.config} is not a JSON file. It must be!"

    return args

def is_dir(paths):
    for path in paths:
        assert os.path.isdir(path), f"{path} is not a directory"

def is_file(paths):
    for path in paths:
        assert os.path.isfile(path), f"{path} is not a directory"


def main():
    args = parse_args()
    root = args.src_dir
    target = args.dst_dir
    
    # Load config json
    with open(args.config,'r') as jsonfile:
        cfg = json.load(jsonfile)

    kwargs= {"extensions": cfg["extensions"]}

    Envisat_DS = Dataset(root,cfg)

    #TODO: Change "p" to "dataset_fraction" and "level" to "intensity_level"

    #Check if Earth augmentation is needed
    Earth_bg = None
    if cfg['corruption_methods']['earth_bg'] is True:
        earth_aug_opts = cfg['opts']['earth_augmentation']
        try:
            ann_file = earth_aug_opts['ann_file']
            earth_bg_dir = earth_aug_opts["earth_bg_images_dir"] 
            Earth_bg = EarthBG(earth_bg_dir, **kwargs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
    Augmented = AugmentedDataset(Envisat_DS, Earth_bg, cfg)
    # This will preserve folder structure filenames while creating the new dataset ;)
    response = Augmented.create_new_dataset(target, **kwargs)

    if response:
        augs = Augmented.assign_augmentations
        log = {'augmentations': augs, 'created': str(datetime.datetime.now()), 'source': root, 'misc': 'Envisat_ICTR'}
        target_fname = 'info_augmentations.json'
        target_json = os.path.join(target,target_fname)
        if os.path.isfile(target_json):
            target_json = os.path.join(target,(target_fname+'_1.json'))

        with open(target_json,'w') as jsonfile:
            json.dump(log,jsonfile, indent=4)

if __name__ == '__main__':
    main()