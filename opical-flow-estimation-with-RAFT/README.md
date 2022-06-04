# RAFT

## Requirements
```Shell
pip3 install -r requirements.txt
```

## How to use

We will use RAFT to create optical flow numpy arrays from two images and save them in a directory. First you will need to download the models. Just run:

```Shell
sh download_models.sh
```

After the download you can run the model on your images like this:

```Shell
run.py --images_dir=<YOUR DIRECTORY> --output_dir=<OUTPUT DIRECTORY>
```
