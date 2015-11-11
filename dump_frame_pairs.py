"""Dump pairs of frames from videos."""

import argparse
import json
import collections
import logging
import random
import sys
import os

from collections import namedtuple

import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('video_list',
                    default=None,
                    help='New-line separated file containing paths to videos.')
parser.add_argument('output',
                    default=None,
                    help='Directory to output frames to.')
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

NUM_PAIRS_PER_VIDEO = 5
PAIR_DISTANCE_SECONDS = 3


class VideoOpenFailedException(Exception):
    pass


class VideoTooShortException(Exception):
    pass

Frame = namedtuple('Frame', ['seconds', 'frame'])
FramePair = namedtuple('FramePair', ['start', 'end'])

# Versions of the above namedtuples that contain information that needs to be
# output to a file on disk for later consumption.
DiskFrame = namedtuple('DiskFrame', ['seconds', 'image_path'])
DiskFramePair = namedtuple('DiskFramePair', ['start', 'end', 'video_path'])

def get_video_info(video_path):
    """Return fps and frame count for a video.

    Theoretically, we can get this from
        video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    but the above does not return a reliable value for large videos.

    Returns:
        video_info (tuple(frame_rate, num_frames))
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise VideoOpenFailedException("Couldn't open video {}".format(
            video_path))
    frame_rate = video.get(cv2.cv.CV_CAP_PROP_FPS)
    video = cv2.VideoCapture(video_path)
    num_frames = 0
    while video.isOpened():
        return_value, frame = video.read()
        if return_value != True:
            break
        num_frames += 1
    video.release()

    return (frame_rate, num_frames)

def capture_frames(video_path, frames_to_capture):
    """Capture specified frames from a video.

    Params:
        video_path (str): Path to a video.
        frames_to_capture (set): Set of frame indices to capture.
    """
    frames_to_capture = set(frames_to_capture)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise VideoOpenFailedException("Couldn't open video {}".format(
            video_path))

    current_frame_index = 0
    captured_frames = dict()
    while video.isOpened():
        return_value, frame = video.read()
        if return_value != True:
            break
        if current_frame_index in frames_to_capture:
            captured_frames[current_frame_index] = frame
        current_frame_index += 1
    video.release()
    return captured_frames


def sample_frame_pairs(video_path, num_pairs_per_video, pair_distance_seconds):
    frame_rate, num_frames = get_video_info(video_path)

    # Sample frame pair indices.
    pair_distance_frames = int(pair_distance_seconds * frame_rate)
    if num_frames <= pair_distance_frames:
        raise VideoTooShortException("Video is less than {} frames.".format(
            pair_distance_frames + 1))
    start_indices = random.sample(
        range(num_frames - pair_distance_frames), num_pairs_per_video)
    start_indices.sort()
    end_indices = [start + pair_distance_frames for start in start_indices]

    frames_to_capture = set(start_indices + end_indices)
    captured_frames = capture_frames(video_path, frames_to_capture)

    # Create FramePair objects.
    frame_pairs = []
    for start_index, end_index in zip(start_indices, end_indices):
        start_seconds = start_index / frame_rate
        start_frame = Frame(seconds=start_seconds,
                            frame=captured_frames[start_index])

        end_seconds = end_index / frame_rate
        end_frame = Frame(seconds=end_seconds,
                          frame=captured_frames[end_index])
        frame_pairs.append(FramePair(start_frame, end_frame))

    return frame_pairs


def main():
    videos = []
    with open(args.video_list) as f:
        videos.extend(f.read().strip().split('\n'))

    # Create output directory; exit if it exists and is not empty.
    if os.path.isdir(args.output):
        if os.listdir(args.output):
            logging.fatal(("Output directory '{}' already exists and is "
                            "not empty.").format(args.output))
            sys.exit(1)
    else:
        os.mkdir(args.output)

    # Store DiskFramePairs to output to disk.
    all_output_frame_pairs = collections.defaultdict(list)
    for video_path in videos:
        try:
            logging.info('Reading video {}'.format(video_path))
            frame_pairs = sample_frame_pairs(video_path, NUM_PAIRS_PER_VIDEO,
                                             PAIR_DISTANCE_SECONDS)
        except (VideoOpenFailedException, VideoTooShortException) as e:
            logging.error(e)
            continue

        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = '{}/{}'.format(args.output, video_basename)
        os.mkdir(video_output_dir)

        # Write frame pairs to disk
        for i, (start_frame, end_frame) in enumerate(frame_pairs):
            pair_output_dir = '{}/pair-{}-{}'.format(video_output_dir,
                                                     int(start_frame.seconds),
                                                     int(end_frame.seconds))
            if os.path.isdir(pair_output_dir): continue
            os.mkdir(pair_output_dir)

            start_image = '{}/start.png'.format(pair_output_dir)
            end_image = '{}/end.png'.format(pair_output_dir)
            cv2.imwrite(start_image, start_frame.frame)
            cv2.imwrite(end_image, end_frame.frame)
            output_frame_pair = DiskFramePair(
                DiskFrame(start_frame.seconds, start_image),
                DiskFrame(end_frame.seconds, end_image), video_path)
            all_output_frame_pairs[video_basename].append(output_frame_pair)
    with open('{}/output.json'.format(args.output)) as f:
        json.dump(all_output_frame_pairs, f)


if __name__ == '__main__':
    main()
