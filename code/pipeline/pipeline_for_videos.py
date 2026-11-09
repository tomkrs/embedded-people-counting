import tensorflow as tf
from sort import Sort
import cv2
import os
import numpy as np
import pipeline_utils as utils

# Inference Pipeline for EfficientDet0-lite + SORT


def run_pipeline(orig_sequence_path: str,
                 sequence_name: str,
                 benchmark_gt_path: str,
                 benchmark_tracker_path: str,
                 frr: float = None,
                 frr_name: str = None,
                 frr_incl: bool = None):

    # Original VIDEO paths to get the sequence, ground truth and seqinfo.ini file
    orig_video_path = orig_sequence_path + '/video/' + sequence_name.upper() + \
        '.mp4'
    orig_sequence_gt_path = orig_sequence_path + '/gt/gt.txt'
    orig_sequence_ini_path = orig_sequence_path + '/seqinfo.ini'

    # CV2 Video Capture Setup
    cap = cv2.VideoCapture(orig_video_path)

    # output tracking file
    sequence_output_file = benchmark_tracker_path + '/' + sequence_name + '.txt'
    if os.path.exists(sequence_output_file):
        os.remove(sequence_output_file)

    # new benchmark root path (only needed for creation of seqinfo.ini file)
    # benchmark_root_path = benchmark_gt_path.replace(sequence_name + '/gt', sequence_name)
    # benchmark_gt_file_path = benchmark_gt_path + '/gt.txt'
    # benchmark_gt_ini_path = benchmark_root_path + '/seqinfo.ini'
    # if os.path.exists(benchmark_gt_file_path):
    #    os.remove(benchmark_gt_file_path)

    # Initialize Object Detection Model
    interpreter = tf.lite.Interpreter(
        model_path="tflite-models\lite-model_efficientdet_lite0_detection_default_1.tflite")
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

    # Only needed for framerate reduction
    # Initialize frame counter
    # orig_frame_counter = 0
    # new_frame_counter = 0

    frame_counter = 0

    # Get the sequence information from the seqinfo.ini file
    # sequence_infos = seqLength, imWidth, imHeight, frameRate
    sequence_infos = utils.get_sequence_info(orig_sequence_ini_path)

    # Calculate the scaling factors
    scale_factor_width = sequence_infos[1] / det_input_width
    scale_factor_height = sequence_infos[2] / det_input_height

    # Person counting setup
    to_left_count = 0
    to_right_count = 0
    prev_x = {}
    centroid_x = 0

    # Iterating over all images in sequence
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video")
            break
        
        frame_counter += 1
        # orig_frame_counter += 1

        # Frame rate reduction
        # if frame is not None:
        #     if frr is not None:
        #         if frr_incl:
        #             if orig_frame_counter % frr == 0:
        #                 frame = None
        #             else:
        #                 new_frame_counter += 1
        #                 utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)
        #         else:
        #             if orig_frame_counter % frr != 0:
        #                 frame = None
        #             else:
        #                 new_frame_counter += 1
        #                 utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)
        #     else:
        #         new_frame_counter += 1
        #         utils.write_new_gT(orig_frame_counter, new_frame_counter, orig_sequence_gt_path, benchmark_gt_file_path)

        # Object Detection
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 320))
            input_data = np.expand_dims(frame_resized, axis=0)

            # set frame as input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # run inference
            interpreter.invoke()

            # Bounding box coordinates of detected objects
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[
                0]  # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[
                0]  # Confidence of detected objects

            person_detected = False
            # Filter the detections to only track the person class and parse it for the SORT tracker
            person_boxes = []
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > det_threshold) and (scores[i] <= 1.0) and (classes[i] == 0)):
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * det_input_height)))
                    xmin = int(max(1, (boxes[i][1] * det_input_width)))
                    ymax = int(
                        min(det_input_height, (boxes[i][2] * det_input_height)))
                    xmax = int(
                        min(det_input_width, (boxes[i][3] * det_input_width)))

                    person_boxes.append(
                        [xmin, ymin, xmax, ymax, scores[i]])
                    person_detected = True

                    # cv2.rectangle(frame_resized, (xmin, ymin),
                    #               (xmax, ymax), (10, 255, 0), 4)

            if person_detected == False:
                person_boxes = np.empty((0, 5))
            else:
                person_boxes = np.array(person_boxes)

            # Update SORT tracker
            track_bbs_ids = sort_tracker.update(person_boxes)

            # Draw the tracking boxes and calculate the centroid of the bounding box for counting
            for track_bb_id in track_bbs_ids:
                x1, y1, x2, y2, track_id = track_bb_id
                cv2.rectangle(frame_resized, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 0), 2)
                # COUNTING
            #     centroid_x = int((x1 + x2) / 2)
            #     if track_id in prev_x:
            #                 # Check if the person has crossed the line
            #                 if prev_x[track_id] < 160 and centroid_x >= 160:
            #                     to_right_count += 1
            #                 elif prev_x[track_id] >= 160 and centroid_x < 160:
            #                     to_left_count += 1
            #             # Update the previous x-coordinate for this person            
            #     prev_x[track_id] = centroid_x
            # with open('code/personcounting/effdet0_lite_sort/' + sequence_name + '.txt', 'a') as f:
            #     f.write(str(frame_counter) + ', ' + str(to_right_count) + ', ' + str(to_left_count) + '\n')
                    
            # Show the frame
            cv2.imshow('Frame', frame_resized)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

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
                    f.write(str(frame_counter) + ',' + str(track_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ', -1 , -1, -1, -1 \n')
        



if __name__ == "__main__":
    video_path = 'Stadtwald_data_for_pipeline'
    
    # for sequence in os.listdir(video_path):
    #     # Print the status of the pipeline
    #     print('Running pipeline for: ' + sequence)
    #     sequence_path = video_path + '/' + sequence
    #     benchmark_gt_path, benchmark_tracker_path = utils.create_new_benchmark_directory(
    #         sequence)
    #     run_pipeline(sequence_path, sequence, benchmark_gt_path,
    #                  benchmark_tracker_path, frr=None, frr_name=None, frr_incl=None)
    
    # Print the status of the pipeline
    sequence = 'x1_y1_z3'
    print('Running pipeline for: ' + sequence)
    sequence_path = video_path + '/' + sequence
    benchmark_gt_path, benchmark_tracker_path = utils.create_new_benchmark_directory(
            sequence)
    run_pipeline(sequence_path, sequence, benchmark_gt_path,
                        benchmark_tracker_path, frr=None, frr_name=None, frr_incl=None)
