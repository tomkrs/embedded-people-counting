# Create directories inside 'code/stadtwald_videos' for the videos to be processed
# 27 videos
# from x1_y1_z1 to x3_y3_z3

import os
import configparser
import cv2
import shutil

def create_seq_info_ini(sequence_name, path):
    config = configparser.ConfigParser()
    
    sequence_ini_path = path + '/seqinfo.ini'
    
    video_path = path + '/video/' + sequence_name.upper() + '.mp4'
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    config['Sequence'] = {
        'name': sequence_name,
        'imDir': 'img1',
        'frameRate': '20',
        'seqLength': total_frames,
        'imWidth': 320,
        'imHeight': 320,
        'imExt': '.jpg'
    }
    with open(sequence_ini_path , 'w') as configfile:
        config.write(configfile)

def copy_hota_gt_files_to_gt_directory():
    for file in os.listdir('code/HOTA_gT'):
        if file.endswith('.txt'):
            source_path = 'code/HOTA_gT/' + file
            sequence_name = file.replace('.txt', '')
            destination_path = 'code/stadtwald_videos/' + sequence_name + '/gt/' + file
            shutil.copy(source_path, destination_path)

def remove_all_gt_videos_in_trackEval():
    stadtwald_gt_path = 'code/results_for_TrackEval/data/gt/mot_challenge/stadtwald-train'
    for sequence_name in os.listdir(stadtwald_gt_path):
        video_file_path = stadtwald_gt_path + '/' + sequence_name + '/video/' + sequence_name.upper() + '.mp4'
        os.remove(video_file_path)
        os.rmdir(stadtwald_gt_path + '/' + sequence_name + '/video')

def rename_all_gt_files_to_gt():
    stadtwald_gt_path = 'code/results_for_TrackEval/data/gt/mot_challenge/stadtwald-train'
    for sequence_name in os.listdir(stadtwald_gt_path):
        original_sequence_gt_filename = stadtwald_gt_path + '/' + sequence_name + '/gt/' + sequence_name + '.txt'
        new_sequence_gt_filename = stadtwald_gt_path + '/' + sequence_name + '/gt/gt.txt'
        os.rename(original_sequence_gt_filename, new_sequence_gt_filename)

def replace_320x320_in_stadtwald_video_filename():
    stadtwald_video_path = 'stadtwald_data_and_scripts/videos'
    for sequence_name in os.listdir(stadtwald_video_path):
        original_sequence_video_filename = stadtwald_video_path + '/' + sequence_name
        new_sequence_video_filename = original_sequence_video_filename.replace('320x320-', '')
        os.rename(original_sequence_video_filename, new_sequence_video_filename)

if __name__ == '__main__':
    replace_320x320_in_stadtwald_video_filename()



