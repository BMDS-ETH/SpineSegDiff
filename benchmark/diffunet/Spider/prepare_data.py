import argparse
import os
import numpy as np
import nibabel as nib




def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./datasets/Dataset008_SPIDER")
    parser.add_argument("-t","--type", type=str, default="3d_data")
    parser.add_argument("-s","--shape", type=int, default=512)
    parser.add_argument("-a","--amount", type=int, help="amount of slices we need", default=32)
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    args = arg_parser()
    data_dir = args.data_dir
    type = args.type
    shape = args.shape
    slices = args.amount
    target_length = args.shape

    os.makedirs(f"{data_dir}_{type}/imagesTr", exist_ok=True)
    os.makedirs(f"{data_dir}_{type}/labelsTr", exist_ok=True)
    print("we are doing something here")
    print(f"we are rescaling to this amount of slices: {slices}")
    print(f'we are having this target shape{slices}{target_length}{target_length}')

    if type == "3d_data" or True:
        all_image_dirs = os.listdir(os.path.join(data_dir,"imagesTr"))
        all_image_paths = [os.path.join(data_dir,"imagesTr",d) for d in all_image_dirs]
        all_seg_dirs = os.listdir(os.path.join(data_dir,"labelsTr"))
        all_seg_paths = [os.path.join(data_dir,"labelsTr",d) for d in all_seg_dirs]
        assert len(all_image_paths) == len(all_seg_paths)


        for i in range(len(all_image_paths)):
            image_path = all_image_paths[i]
            seg_path = all_seg_paths[i]

            image_save_path = image_path.replace(f"{data_dir}",f"{data_dir}_{type}")
            seg_save_path = seg_path.replace(f"{data_dir}",f"{data_dir}_{type}")
            image_data = nib.load(image_path).get_fdata()
            seg_data = nib.load(seg_path).get_fdata()
            assert image_data.shape == seg_data.shape

            current_slices = image_data.shape[0]
            curent_length = image_data.shape[1]
            
            
            #select only the right amount of slices
            new_image = image_data[current_slices//2 - slices//2 : current_slices//2 + slices//2,
                                   curent_length//2 - target_length//2:curent_length//2 +target_length//2,
                                   curent_length//2 - target_length//2:curent_length//2 +target_length//2]
            
            new_seg = seg_data[current_slices//2 - slices//2 : current_slices//2 + slices//2,
                               curent_length//2 - target_length//2:curent_length//2 +target_length//2,
                               curent_length//2 - target_length//2:curent_length//2 +target_length//2]


            nib.save(nib.Nifti1Image(new_image,affine=np.eye(4)),image_save_path)
            nib.save(nib.Nifti1Image(new_seg,affine=np.eye(4)),seg_save_path)
        print(image_data.shape, "shape of old image")
        print(new_image.shape, "shape of new image")
