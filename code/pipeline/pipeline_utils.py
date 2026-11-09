import configparser
import os



def get_sequence_info(seqinfo_ini_path: str):
    '''
    Returns the sequence information from the seqinfo.ini file as a tuple
    :param seqinfo_ini_path: path to the seqinfo.ini file
    :return: tuple of (seqLength, imWidth, imHeight, frameRate)
    '''
    config = configparser.ConfigParser()
    config.read(seqinfo_ini_path)
    sequence_info = config['Sequence']
    return (int(sequence_info['seqLength']),
            int(sequence_info['imWidth']),
            int(sequence_info['imHeight']),
            float(sequence_info['frameRate']))

def write_new_gT(orig_frame_counter: int, new_frame_counter: int, sequence_gt_path: str, benchmark_gt_file_path: str):
    '''
    Writes the new ground truth file for the benchmark
    :param orig_frame_counter: original frame counter
    :param new_frame_counter: new frame counter
    :param sequence_gt_path: path to the original ground truth file
    :param benchmark_gt_path: path to the new ground truth file
    :return:
    '''
    lines_to_write = []
    with open(sequence_gt_path, 'r') as sequence_gt_file:
        for line in sequence_gt_file:
            split_line = line.split(',')
            if split_line[0] == str(orig_frame_counter):
                split_line[0] = str(new_frame_counter)
                new_line = ','.join(split_line)
                lines_to_write.append(new_line)
    with open(benchmark_gt_file_path, 'a') as new_gt:
        for line in lines_to_write:
            new_gt.write(line)

def create_new_gT_seqinfo_ini(sequence_infos, benchmark_gt_ini_path, sequence_name, new_frame_counter):
    '''
    Creates the new seqinfo.ini file for the sequence with the new frame counter
    :param sequence_infos: tuple of (seqLength, imWidth, imHeight, frameRate)
    :param benchmark_gt_ini_path: path to the new seqinfo.ini file
    :param new_frame_counter: new frame counter
    :param sequence_name: name of the sequence
    :return:
    '''
    config = configparser.ConfigParser()
    new_sequence_length = new_frame_counter
    new_framerate = int((int(sequence_infos[3]) / int(sequence_infos[0])) * new_frame_counter)
    config['Sequence'] = {
        'name': sequence_name,
        'imDir': 'img1',
        'frameRate': str(new_framerate),
        'seqLength': str(new_sequence_length),
        'imWidth': str(sequence_infos[1]),
        'imHeight': str(sequence_infos[2]),
        'imExt': '.jpg'
    }
    with open(benchmark_gt_ini_path, 'w') as configfile:
        config.write(configfile)
    
    

def create_new_benchmark_directory(sequence, mot=None, frr_name=None):
    '''
    Creates the new benchmark directory for the given sequence
    :param mot: mot challenge
    :param frr_name: name of the frr
    :param sequence: sequence name
    :return: benchmark_gt_path, benchmark_tracker_path
    '''
    if mot is None and frr_name is None:
        benchmark_gt_path = 'code/results_for_TrackEval/data/gt/mot_challenge/stadtwald-train/' + sequence + '/gt'
        os.makedirs(benchmark_gt_path, exist_ok=True)
        benchmark_tracker_path = 'code/results_for_TrackEval/data/trackers/mot_challenge/stadtwald-train/EffDet0lite_SORT/data'
        os.makedirs(benchmark_tracker_path, exist_ok=True)
    else:
        benchmark_gt_path = 'code/results_for_TrackEval/data/gt/mot_challenge/' + mot + '_' + frr_name + '-train' + '/' + sequence + '/gt'
        os.makedirs(benchmark_gt_path, exist_ok=True)
        benchmark_tracker_path = 'code/results_for_TrackEval/data/trackers/mot_challenge/' + mot + '_' + frr_name + '-train' + '/EffDet0lite_SORT/data'
        os.makedirs(benchmark_tracker_path, exist_ok=True)
    return benchmark_gt_path, benchmark_tracker_path



def create_seqmaps(mot_path, mot=None, frr_name=None):
    '''
    Creates the seqmaps file for the benchmark 
    :param mot: mot challenge
    :param frr_name: name of the frr
    :param mot_path: path to the mot challenge
    :return:
    '''
    if mot is None and frr_name is None:
        seqmap_file_path = 'code/results_for_TrackEval/data/gt/mot_challenge/seqmaps/stadtwald-train.txt'
        with open(seqmap_file_path, 'w') as seqmap_file:
            seqmap_file.write('name\n')
            for sequence in os.listdir(mot_path):
                seqmap_file.write(sequence + '\n')
    else:    
        seqmap_file_path = 'code/results_for_TrackEval/data/gt/mot_challenge/seqmaps/' + mot + '_' + frr_name + '-train.txt'
        with open(seqmap_file_path, 'w') as seqmap_file:
            seqmap_file.write('name\n')
            for sequence in os.listdir(mot_path):
                if sequence != 'KITTI-13' and sequence != 'KITTI-17':
                    seqmap_file.write(sequence + '\n')