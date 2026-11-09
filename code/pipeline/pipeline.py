import tensorflow as tf
from sort import Sort
import cv2
import os
import numpy as np
import pipeline_utils as utils

# Inference Pipeline for EfficientDet0-lite + SORT

def run_pipeline(orig_sequence_path: str, 
                    sequence_name: str,
                    benchmark_gt_path:str,
                    benchmark_tracker_path:str,
                    frr: float = None, 
                    frr_name: str = None, 
                    frr_incl: bool = True):
    
    # Original MOT paths to get the images, ground truth and seqinfo.ini file
    orig_sequence_images_path = orig_sequence_path + '/img1/'
    orig_sequence_gt_path = orig_sequence_path + '/gt/gt.txt'
    orig_sequence_ini_path = orig_sequence_path + '/seqinfo.ini'

    # output tracking file
    sequence_output_file = benchmark_tracker_path + '/' + sequence_name + '.txt'
    if os.path.exists(sequence_output_file):
        os.remove(sequence_output_file)

    # new benchmark root path (only needed for creation of seqinfo.ini file)
    benchmark_root_path = benchmark_gt_path.replace(sequence_name + '/gt', sequence_name)
    benchmark_gt_file_path = benchmark_gt_path + '/gt.txt'
    benchmark_gt_ini_path = benchmark_root_path + '/seqinfo.ini'
    if os.path.exists(benchmark_gt_file_path):
        os.remove(benchmark_gt_file_path)

    # Initialize Object Detection Model
    interpreter = tf.lite.Interpreter(model_path="tflite-models\lite-model_efficientdet_lite0_detection_default_1.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    det_input_width = input_details[0]['shape'][2]
    det_input_height = input_details[0]['shape'][1]

    det_threshold = 0.5

    # Initialize SORT Tracker
    # Todo: GridSearch for best hyperparameters
    # max_age= 100 und min_hits=1 wie bei SORT20 https://motchallenge.net/method/MOT=3230&chl=13
    sort_tracker = Sort(max_age=100, min_hits=1)

    # Initialize frame counter
    orig_frame_counter = 0
    new_frame_counter = 0

    # Get the sequence information from the seqinfo.ini file
    # sequence_infos = seqLength, imWidth, imHeight, frameRate
    sequence_infos = utils.get_sequence_info(orig_sequence_ini_path)

    # Calculate the scaling factors
    scale_factor_width = sequence_infos[1] / det_input_width
    scale_factor_height = sequence_infos[2] / det_input_height

    
    # Iterating over all images in sequence
    for image in os.listdir(orig_sequence_images_path):
        if image.endswith(".jpg"):
            image_path = orig_sequence_images_path + image
            frame = cv2.imread(image_path)
            orig_frame_counter += 1

            # Frame rate reduction
            if frame is not None:
                if frr is not None:
                    if frr_incl:
                        if orig_frame_counter % frr == 0:
                            frame = None
                        else:
                            new_frame_counter += 1                   
                            utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)
                    else:
                        if orig_frame_counter % frr != 0:
                            frame = None
                        else:
                            new_frame_counter += 1     
                            utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)
                else:
                    new_frame_counter += 1     
                    utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)

            # Object Detection
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (320, 320))
                input_data = np.expand_dims(frame_resized, axis=0)

                # set frame as input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # run inference
                interpreter.invoke()
                    
                boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

                person_detected = False
                # Filter the detections to only track the person class and parse it for the SORT tracker
                person_boxes = []
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > det_threshold) and (scores[i] <= 1.0) and (classes[i] == 0)):
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1, (boxes[i][0] * det_input_height)))
                        xmin = int(max(1, (boxes[i][1] * det_input_width)))
                        ymax = int(min(det_input_height, (boxes[i][2] * det_input_height)))
                        xmax = int(min(det_input_width, (boxes[i][3] * det_input_width)))

                        person_boxes.append([xmin, ymin, xmax, ymax, scores[i]])
                        person_detected = True

                        cv2.rectangle(frame_resized, (xmin, ymin),
                                    (xmax, ymax), (10, 255, 0), 4)
                        
                if person_detected == False:
                    person_boxes = np.empty((0,5))
                else:
                    person_boxes = np.array(person_boxes)
                
                # Update SORT tracker
                track_bbs_ids = sort_tracker.update(person_boxes)
                
                # Draw the tracking boxes
                for track_bb_id in track_bbs_ids:
                    x1, y1, x2, y2, track_id = track_bb_id
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Parse the tracking result for the HOTA TrackEvaluation Tool
                # Format: frame_number, track_id, x1, y1, width, height, conf_score, a, b, c
                # a, b, c are not used in the evaluation -> set to -1
                for track_bb_id in track_bbs_ids:
                    x1, y1, x2, y2, track_id = track_bb_id
                    # Rescale the coords to the original image size
                    x1 = x1 * scale_factor_width
                    y1 = y1 * scale_factor_height
                    x2 = x2 * scale_factor_width
                    y2 = y2 * scale_factor_height

                    # Write tracking result to file for HOTA evaluation
                    with open(sequence_output_file, 'a') as f:
                        f.write(str(new_frame_counter) + ',' + str(track_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2-x1) + ',' + str(y2-y1) + ', -1 , -1, -1, -1 \n')
    
    # Create the new seqinfo.ini file for the benchmark
    utils.create_new_gT_seqinfo_ini(sequence_infos, benchmark_gt_ini_path, sequence_name, new_frame_counter)

if __name__ == "__main__":
    
    frame_inclusion_list = [i for i in range(3, 21)] # Behalte alle bis auf den n-ten Frame
    frame_inclusion_list.reverse()
    frame_exclusion_list = [i for i in range(2, 21)] # Behalte nur das n-te Frame
    frame_inclusion_list_strings = ['keepNotFrame_' + str(i) for i in frame_inclusion_list]
    frame_exclusion_list_strings = ['keepOnlyFrame_' + str(i) for i in frame_exclusion_list]
    frame_none_liste = [None]
    frame_none_liste_strings = ['keepAllFrames']

    mot_list = ['MOT16']
    for mot in mot_list:
        mot_path = 'code/MOT_data/' + mot
        print('Running pipeline for: ' + mot)
        
        for frr_incl, frr_incl_name in zip(frame_inclusion_list, frame_inclusion_list_strings):
            for sequence in os.listdir(mot_path):
                if sequence != 'KITTI-13' and sequence != 'KITTI-17':
                    # Print the status of the pipeline
                    print('Running pipeline for: ' + mot + ' ' + frr_incl_name + ' ' + sequence)
                    sequence_path = mot_path + '/' + sequence
                    benchmark_gt_path, benchmark_tracker_path = utils.create_new_benchmark_directory(sequence, mot, frr_incl_name)
                    run_pipeline(sequence_path, sequence, benchmark_gt_path, benchmark_tracker_path, frr=frr_incl, frr_name=frr_incl_name, frr_incl=True)
            utils.create_seqmaps(mot_path, mot, frr_incl_name)
        
        for frr_excl, frr_excl_name in zip(frame_exclusion_list, frame_exclusion_list_strings):
            print('Running pipeline for: ' + mot + ' ' + frr_excl_name)
            for sequence in os.listdir(mot_path):
                if sequence != 'KITTI-13' and sequence != 'KITTI-17':
                    print('Running pipeline for: ' + mot + ' ' + frr_excl_name + ' ' + sequence)
                    benchmark_gt_path, benchmark_tracker_path = utils.create_new_benchmark_directory(sequence, mot, frr_excl_name)
                    sequence_path = mot_path + '/' + sequence
                    run_pipeline(sequence_path, sequence, benchmark_gt_path, benchmark_tracker_path, frr=frr_excl, frr_name=frr_excl_name, frr_incl=False)
            utils.create_seqmaps(mot_path, mot, frr_excl_name)
        print('Finished pipeline for: ' + mot)
        print('------------------------------------------------------------------------------------------------------------------')
        
        for frr_none, frr_none_name in zip(frame_none_liste, frame_none_liste_strings):
                for sequence in os.listdir(mot_path):
                    if sequence != 'KITTI-13' and sequence != 'KITTI-17':
                        print('Running pipeline for: ' + mot + ' ' + frr_none_name + ' ' + sequence)
                        benchmark_gt_path, benchmark_tracker_path = utils.create_new_benchmark_directory(sequence, mot, frr_none_name)
                        sequence_path = mot_path + '/' + sequence
                        run_pipeline(sequence_path, sequence, benchmark_gt_path, benchmark_tracker_path)
                utils.create_seqmaps(mot_path, mot, frr_none_name)
        print('Finished pipeline for: ' + mot)
        print('------------------------------------------------------------------------------------------------------------------')
        

