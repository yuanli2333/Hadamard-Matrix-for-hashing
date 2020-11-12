import os
import csv
import logging
import subprocess
from os import listdir

from joblib import delayed
from joblib import Parallel

def exe_cmd(cmd):
    try:
        dst_file = cmd.split()[-1]
        if os.path.exists(dst_file):
            return "exist"
        cmd = cmd.replace('(', '\(').replace(')', '\)').replace('\'', '\\\'')
        output = subprocess.check_output(cmd, shell=True,
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logging.warning("failed: {}".format(cmd))
        # logging.warning("failed: {}: {}".format(cmd, err.output.decode("utf-8"))) # detailed error
        return False
    return output


scr_root = 'data'
dst_root = 'nvvl_data_avi'
if not os.path.exists(dst_root):
    os.makedirs(dst_root)

cmd_format = 'ffmpeg -i {} -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high {}.mp4'
#commands = []
in_parallel = False
for f in listdir(scr_root):
    commands = []
    sub_root = scr_root + '/'+ f +'/'
    print('processing in: %s'%sub_root)
    #scr_root = os.path.join(scr_root, f)
    output_root_dir = os.path.join(dst_root, f)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    for sub_name in listdir(sub_root):
        #print('processing: %s'%sub_name)
        basename = os.path.splitext(sub_name)[0]
        #print(basename)
        input_video_path = os.path.join(sub_root,sub_name)
        #print(input_video_path)
        output_root_prefix = output_root_dir+'/'+basename
        #print(output_video_path)
        
        cmd = cmd_format.format(input_video_path, output_root_prefix)
        commands.append(cmd)

    num_jobs = 8
    logging.info("processing videos in parallel, num_jobs={}".format(num_jobs))
    Parallel(n_jobs=num_jobs)(delayed(exe_cmd)(cmd) for cmd in commands)
