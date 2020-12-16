import os
import argparse
import gdown
import hashlib
import json

# parse args
parser = argparse.ArgumentParser(description='Download files for HOPE datasets.')
parser.add_argument('--overwrite', action='store_true',
                    help='Download and overwrite existing paths')
parser.add_argument('--skip-eval-meshes', action='store_true',
                    help='Omit low-res object meshes')
parser.add_argument('--skip-hope-image', action='store_true',
                    help='Omit HOPE-Image dataset')
parser.add_argument('--skip-hope-video', action='store_true',
                    help='Omit HOPE-Video dataset')
parser.add_argument('--skip-test-sets', action='store_true',
                    help='Omit test sets and only download validation sets')
parser.add_argument('--do-test', action='store_true',
                    help='Debug')
args = parser.parse_args()

# read list of urls for downloading the dataset
urls = json.load(open('download_urls.json'))
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
if args.do_test:
    download_and_extract('test', skip_existing=not args.overwrite)

if not args.skip_eval_meshes:
    download_and_extract(
        'eval_meshes',
        msg='low-res eval meshes',
        skip_existing=not args.overwrite)

if not args.skip_hope_image:
    download_and_extract(
        'hope_image_val', 
        msg='HOPE-Image validation set',
        skip_existing=not args.overwrite
    )

if not args.skip_hope_image and not args.skip_test_sets:
    download_and_extract(
        'hope_image_test', 
        msg='HOPE-Image test set',
        skip_existing=not args.overwrite
    )

if not args.skip_hope_video:
    download_and_extract(
        'hope_video_val', 
        msg='HOPE-Video validation set',
        skip_existing=not args.overwrite
    )

if not args.skip_hope_video and not args.skip_test_sets:
    download_and_extract(
        'hope_video_test', 
        msg='HOPE-Video test set',
        skip_existing=not args.overwrite
    )
