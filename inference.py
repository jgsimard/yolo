from argparse import ArgumentParser
from network.detector import Detector


def main():
    parser = ArgumentParser(description=' Test the detector ')
    parser.add_argument('-file_path', metavar=None, help='Test the detector on a media file, give the path')
    parser.add_argument('-camera', action='store_true', help='Test the detector on the live feed of the camera')
    parser.add_argument('-test_img', action='store_true', help='Test the detector on sample images')
    parser.add_argument('-test_video', action='store_true', help='Test the detector on a sample video')
    parser.add_argument('-save', action='store_true', help='Save the results')
    parser.add_argument('-alive', action='store_true', help='Wait for files to analyse')
    args = parser.parse_args()

    launch_detector(args.camera, args.test_img, args.test_video, args.file_path, args.alive, args.save)


def launch_detector(camera, test_img, test_video, file_path, alive, save):
    if camera or test_img or test_video or file_path or alive:
        detector = Detector(save)
        if camera:
            detector.camera()
        if test_img:
            detector.test_images()
        if test_video:
            detector.test_videos()
        if file_path:
            detector.file(args.file_path)
        if alive:
            detector.alive()
    else:
        print('Nothing to do \nUse -h or --help for more information')


if __name__ == '__main__':
    main()
