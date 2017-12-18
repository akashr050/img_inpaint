Dependencies:
Tensorflow 1.4
Python 2

Steps:
1. Download and install the dependencies.
2. Make folders named checkpoints and tb_results in the root directory.
3. Download the dataset from the CelebA website (https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACA-mNX0tHOAfPLuOCmNfe7a/Img/img_align_celeba_png.7z?dl=0). We are using the img_align_celeba images from the dataset rather than the raw images.
4. Generate the eval.txt and train.txt using the file generate_train_eval.py after changing the respective directories.
5. Place the dataset img_align_celeba in the raw_images folder in the workspace folder in the root directory.
6. Train the network by running the train_glc.py.
7. The checkpoints are saved in the checkpoints folder.
8. The tensorboard results are stored in the tb_results folder.
