import os
import shutil
import subprocess
if __name__ == '__main__':
    subprocess.run(["cd", "dataset"], shell=True)
    subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"], shell=True)
    subprocess.run(["unzip", "tiny-imagenet-200.zip -d tiny-imagenet"])
    
    with open('dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')