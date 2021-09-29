import glob
import json
import os
import random
import datetime
import cv2
import numpy as np
from PIL import Image
#from pycocotools.coco import COCO # Only for COCO dataset

from augmentations import Augmentations, FileHelpers

#get_ipython().run_line_magic('matplotlib', 'inline')


class EarthBG:

    def __init__(self, img_dir, **kwargs):
        self.img_dir = img_dir
        extension = kwargs['extensions']['background']
        self.bg_files = glob.glob(os.path.join(img_dir,f'*.{extension}'))
        self.shape_pts = {
            'body': list(range(4)) + list(range(8,12)),
            'antenna':list(range(4,8)),
            'panel': [12,13,15,14]
        }

    def get_shape_pts(self, idxs, kps):
        shape_pts = np.zeros((len(idxs),2),dtype='int32')
        for k,idx in enumerate(idxs):
            shape_pts[k,:] = kps[idx]
        return shape_pts

    def get_random_bg_snip(self, snip_size = (1000,1000), target_size = (512,512)):
        random.shuffle(self.bg_files)
        select_bg = self.bg_files[0]
        bg = cv2.imread(select_bg)

        bg_h0 = bg.shape[0]
        bg_w0 = bg.shape[1]
        delta = 50
        h_lim = (delta, bg_h0-snip_size[0]-delta)
        w_lim = (delta, bg_w0-snip_size[1]-delta)

        y0 = random.randint(h_lim[0], h_lim[1])
        x0 = random.randint(w_lim[0], w_lim[1])

        cropped = bg[y0:y0+snip_size[0], x0:x0+snip_size[1]]

        resized = cv2.resize(cropped,target_size)
        return resized

    def add_earth_to_img(self, foreground, kps, fov = (1000,1000), blur_bg = (3,3), blur_kernel_mask=(1,1)):
        '''
        foreground: pil image
        kps: list of envisat keypoints
        fov: snip size - approximate fov in Himawari image from cam altitude

        '''
        fg = np.array(foreground)
        fg2 = fg.copy()

        body_pts0 = self.get_shape_pts(self.shape_pts['body'], kps)
        panel_pts = self.get_shape_pts(self.shape_pts['panel'], kps)
        antenna_pts = self.get_shape_pts(self.shape_pts['antenna'], kps)

        #find external points to create mask. order with convexhull. Others are ordered in shapes method
        body_pts = cv2.convexHull(body_pts0)

        #create masks. Successive
        panel_mask = cv2.fillConvexPoly(fg2,panel_pts , color= (255, 255, 255))
        body_mask = cv2.fillConvexPoly(fg2,body_pts , color= (255, 255, 255))
        antenna_mask = cv2.fillConvexPoly(fg2,antenna_pts , color= (255, 255, 255))

        full_mask = antenna_mask.copy()

        # Morphological Transformation nad blur
        mask0 = cv2.dilate(full_mask, None, iterations=1)
        mask0 = cv2.erode(mask0, None, iterations=1)
        fg_mask_blurred= (cv2.GaussianBlur(mask0, blur_kernel_mask, 0)).astype(np.uint8)
        fg_masked = cv2.bitwise_and(fg,fg_mask_blurred)

        #Background Operations
        bg = self.get_random_bg_snip(snip_size=fov, target_size=(512,512))
        bg_blurred = cv2.GaussianBlur(bg,blur_bg,0)

        #mask_not_sat= cv2.bitwise_not(cv2.GaussianBlur(fg_mask_blurred,blur_kernel_mask,0))
        mask_not_sat= cv2.bitwise_not(fg_mask_blurred)
        bg_mask = cv2.bitwise_and(mask_not_sat, bg_blurred)
        bg_masked = cv2.GaussianBlur(bg_mask, blur_kernel_mask, 0)

        augmented = cv2.bitwise_or(fg_masked,bg_masked)

        return augmented


class AnnotatedDataset:
    def __init__ (self, img_dir, json_file):
        self.anns,self.filenames = self.loadAnns(json_file)
        self.index_map = self.get_filename_indexmap(self.anns)
        self.subfolders, self.subfolder_to_idx = FileHelpers.find_subfolders(img_dir)
        self.imgs = FileHelpers.make_dataset(img_dir, self.subfolder_to_idx)

    def loadAnns(self, json_file):
        with open(json_file,'r') as jsonf:
            dicts = json.load(jsonf)

        filenames = [ ann['filename'] for ann in dicts]
        return dicts, filenames

    def get_filename_indexmap(self, anns):
        index_map = {}
        for idx, ann in enumerate(anns):
            index_map[ann['filename']]= idx
        return index_map
    '''
    def check_extension(self, filename, add = False, remove = False):
        split_ext =  filename.split('.')
        ext = ''
        if len(split_ext)>1:
            ext = split_ext[-1]
        if add:
            ext = ext if ext!='' else 'jpg'
            return f"{filename}.{ext}", True
        elif remove:
            return split_ext[0], False
        else:
            exists = False if ext =='' else True
            return filename, exists
    '''
    def get_name_from_id(self,id):
        return f"{self.imgname_header}{id}.jpg"

    def get_id_from_name(self, name):
        split_ext =  name.split('.')
        img_id = int(split_ext[0].split(self.filename_header)[-1])
        return img_id

    def get_name_from_path(self, filepath):
        path_type1 = filepath.split('/')
        path_type2 = filepath.split('\\')
        if len(path_type1)>1:
            return path_type1[-1]
        else:
            return path_type2[-1]


class Dataset:
    def __init__ (self, img_dir, cfg):
        self.extension = cfg['extensions']['original']
        self.filepaths = glob.glob(os.path.join(img_dir,f'*.{self.extension}')) #TODO: Add subfolder walk ?
        self.filenames, self.index_map = self.get_filename_indexmap(self.filepaths)
        self.subfolders, self.subfolder_to_idx = FileHelpers.find_subfolders(img_dir)
        self.imgs = FileHelpers.make_dataset(img_dir, self.subfolder_to_idx)

    def get_filename_indexmap(self, filepaths):
        index_map = {}
        filenames = []
        for idx,filepath in enumerate(filepaths):
            filename= filepath.split('/')[-1]
            filenames.append(filename)
            index_map[filename]= idx
        return filenames, index_map



class AugmentedDataset:
    
    def __init__(self, source_DS, Earth_DS, cfg):# , augmentations):
        
        #source_DS - type Dataset
        #Earth_DS - type EarthBG class
        self.methods= Augmentations.methods 
        self.augs_list= Augmentations.get_list()

        self.source_DS = source_DS
        self.filenames = source_DS.filenames
        self.img_tuples = source_DS.imgs

        self.layers = cfg["ordered_randomization_layers"]
        ordered_effects_list = [self.layers[layer]["effects"] for layer in list(self.layers)]
        self.layer_map = self.get_layer_effect_map(list(self.layers), ordered_effects_list)

        self.levels = {key: value['level'] for key,value in self.layers.items()}
        self.prob = {key: value['p'] for key,value in self.layers.items()}

        if Earth_DS is not None and isinstance(Earth_DS, EarthBG): 
            self.augs_list.append('earth_bg')
            self.methods['earth_bg'] = Earth_DS.add_earth_to_img
            self.prob['background'] = self.layers["background"]["p"]        #train- 0.2, val- 0.2

        self.assign_augmentations = {}
        for filename in self.filenames:
            self.assign_augmentations[filename] = []

        self.assign_one_random()

        self.subfolders = source_DS.subfolders
        self.subfolder_to_idx = source_DS.subfolder_to_idx

    def get_layer_effect_map(self, keys, values):
        '''
        keys:list
        values: list / list(list)

        Values must be a single value or a list
        '''

        assert (isinstance(keys, list) and isinstance(values, list) and len(keys)==len(values)), "Invalid input. keys and values should be lists with equal number of elements"
        inverted_dict = {}
        for idx,key in enumerate(keys):
            effects = values[idx]
            if isinstance(effects,list):
                for effect in effects:
                    inverted_dict[effect] = key 
        return inverted_dict



    def assign_one_random(self):
        all_augs = self.augs_list
        for img_name in self.filenames:
            random.shuffle(all_augs)
            thisaug= all_augs[0]
            aug_layer = self.layer_map[thisaug]
            thislevel = self.levels[aug_layer]
            self.assign_augmentations[img_name] = [{"method":thisaug, 'level':thislevel}]

    def assign_earth_bg_and_random_erase(self):
        
        #each should be applied on different images. cannot be on the same image
        img_names = self.filenames
        random.shuffle(img_names)
        n_earth_bg = int(self.prob['background']*(len(img_names)))
        n_erase = int(self.prob['erase']*(len(img_names)))

        if n_earth_bg != 0:
            selected_earth_bg = img_names[:n_earth_bg]
            self.update_augmentations(selected_earth_bg,{'method':'earth_bg', 'level':None})
        
        if n_earth_bg != 0:
            selected_random_erase = img_names[-n_erase:]
            self.update_augmentations(selected_random_erase,{'method':'random_erase', 'level':None})

    def assign_color_jitter(self):
        
        img_names = self.filenames
        random.shuffle(img_names)
        n_jitter = int(self.prob['color']*(len(img_names)))
        
        if n_jitter!=0:
            selected = img_names[:n_jitter]
            self.update_augmentations(selected,{'method':'color_jitter', 'level':None})

    def assign_blur(self, level=1):

        img_names = self.filenames
        random.shuffle(img_names)
        n_augment = int(self.prob['blur']*(len(img_names)))
        blur_effects = self.layers['blur']['effects']

        for idx, img_name in enumerate(img_names):
            if idx>n_augment:
                break
            effect = blur_effects[random.randrange(0,len(blur_effects))]

            self.update_augmentations([img_name],{'method':effect, 'level':level})

    def assign_noise(self, level=1):

        img_names = self.filenames
        random.shuffle(img_names)
        n_augment = int(self.noise_prob['noise']*(len(img_names)))
        noise_effects = self.layers['noise']['effects']

        for idx, img_name in enumerate(img_names):
            if idx>n_augment:
                break
            effect = noise_effects[random.randrange(0,len(noise_effects))]
            self.update_augmentations([img_name],{'method':effect, 'level':level})


    def update_augmentations(self, keys, value):
        for key in keys:
            self.assign_augmentations[key].append(value)

    def apply_all_augmentations(self, img, fname, aug_list):
        this_img = img.copy()
        for aug in aug_list:
            method = aug['method']
            if method == 'earth_bg':
                this_img = self.add_earth_augmentation(this_img, fname)
            else:
                if aug['level']:
                    this_img = self.methods[method](this_img,aug['level'])
                else:
                    this_img = self.methods[method](this_img)
        return this_img


    def add_earth_augmentation(self, img, fname):
        ann = self.source_DS.anns[self.source_DS.index_map[fname]]
        kps_info = ann['features']
        kps = [d['Coordinates'] for d in kps_info ]
        bg_augmented = self.methods['earth_bg'](img,kps)
        return bg_augmented



    def create_new_dataset(self, target_dir, **kwargs):

        for fpath, fname, subfolder_idx in self.img_tuples:
            print(f"processing {fname}")
            if os.path.isfile(fpath):
                img= cv2.imread(fpath)
                augs = self.assign_augmentations[fname]
                save_path = target_dir
                if self.subfolders:
                    target_subfolder = self.subfolders[subfolder_idx]
                    save_path = os.path.join(save_path,target_subfolder)
                os.makedirs(save_path, exist_ok=True)
                target_file = os.path.join(save_path,fname)

                #if kwargs['skip_existing'] and os.path.isfile(target_file):
                #    print(f"Skipped Existing augmented image: {target_file}")
                #else:
                #    img_transformed = self.apply_all_augmentations(img, fname, augs)
                #    cv2.imwrite(target_file, img_transformed)

                img_transformed = self.apply_all_augmentations(img, fname, augs)
                cv2.imwrite(target_file, img_transformed)
            else:
                print(f"ERROR (does not exist) :  {fname} \n File augmentation skipped")

        return True


