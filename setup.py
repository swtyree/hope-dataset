import os
import argparse
import hashlib
import json

try:
    import gdown
except ModuleNotFoundError as e:
    print(f'The Python package `gdown` is required to download the dataset.\nPlease run: `pip install gdown`\n')
    raise


# parse args
parser = argparse.ArgumentParser(description='Download files for HOPE datasets.')
parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing paths')

parser.add_argument('--meshes', action='store_true',
                    help='Download low-res and high-res object meshes')
parser.add_argument('--meshes-eval', action='store_true',
                    help='Download low-res object meshes')
parser.add_argument('--meshes-full', action='store_true',
                    help='Download high-res object meshes')

parser.add_argument('--image', action='store_true',
                    help='Download HOPE-Image dataset')
parser.add_argument('--image-valid', action='store_true',
                    help='Download HOPE-Image validation dataset')
parser.add_argument('--image-test', action='store_true',
                    help='Download HOPE-Image test dataset')

parser.add_argument('--video', action='store_true',
                    help='Download HOPE-Video dataset')
parser.add_argument('--video-valid', action='store_true',
                    help='Download HOPE-Video validation dataset')
parser.add_argument('--video-test', action='store_true',
                    help='Download HOPE-Video test dataset')

args = parser.parse_args()

# by default, download all parts
if not any([
    args.meshes, args.meshes_eval, args.meshes_full, 
    args.image, args.image_valid, args.image_test, 
    args.video, args.video_valid, args.video_test
]):
    args.meshes = True
    args.image = True
    args.video = True

if args.meshes:
    args.meshes_eval = True
    args.meshes_full = True

if args.image:
    args.image_valid = True
    args.image_test = True

if args.video:
    args.video_valid = True
    args.video_test = True

# read list of urls for downloading the dataset
urls = json.load(open('setup.json'))
filter_urls = lambda k: [(u['url'], u['dest'], u['md5']) for u in urls if u['group']==k]

# function to compute md5 hashes (why isn't this already in hashlib?!)
# (from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python)
def compute_md5(fn, BUF_SIZE=65536):
    md5 = hashlib.md5()
    with open(fn, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()    

# function to download and extract archives
def download_and_extract(group, msg=None, skip_existing=True):
    filtered_urls = filter_urls(group)
    print(f'Downloading {group if msg is None else msg} ({len(filtered_urls)} file{"s" if len(filtered_urls) != 1 else ""})...\n')
    for url,dest,md5 in filtered_urls:
        # skip if path already exists
        if os.path.exists(dest) and skip_existing:
            print(f'Path {dest} exists; skipping.\n(To not skip, use option --overwrite.)')
            continue
        
        # download archive file and check md5 hash
        fn = gdown.download(url=url, quiet=False)
        assert compute_md5(fn)==md5, 'Downloaded file failed MD5 hash! Exiting...'
        print('MD5 passed.')
        
        # make target path, extract, and delete archive
        os.makedirs(dest, exist_ok=True)
        gdown.extractall(path=fn, to=dest)
        os.remove(fn)
        print('Extracted.')
    print('\nDone.\n\n')

# download requested parts of dataset
if args.meshes_eval:
    download_and_extract(
        'meshes_eval',
        msg='low-res eval meshes',
        skip_existing=not args.overwrite)

if args.meshes_eval:
    download_and_extract(
        'meshes_full',
        msg='full-res meshes',
        skip_existing=not args.overwrite)

if args.image_valid:
    download_and_extract(
        'hope_image_valid', 
        msg='HOPE-Image validation set',
        skip_existing=not args.overwrite
    )

if args.image_test:
    download_and_extract(
        'hope_image_test', 
        msg='HOPE-Image test set',
        skip_existing=not args.overwrite
    )

if args.video_valid:
    download_and_extract(
        'hope_video_valid', 
        msg='HOPE-Video validation set',
        skip_existing=not args.overwrite
    )

if args.video_test:
    download_and_extract(
        'hope_video_test', 
        msg='HOPE-Video test set',
        skip_existing=not args.overwrite
    )
