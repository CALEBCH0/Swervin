import argparse

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')

    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '--config', type=str, default="config.toml",
    )
    
    parser.add_argument(
        '-p', '--profile', action='store_true',
        help='Enable performance profiling')
    
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help='Enable benchmarking mode'
    )
    
    parser.add_argument(
        '-r', '--record', type=str, default='default',
        help='Set session name for recorder'
    )
    
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')

    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')

    parser.add_argument(
        '-ir', '--ignore_recording', action='store_true',
        help='Ignore Recording')

    parser.add_argument(
        '-s', '--sound_mode',
        default='silent',
        type=str,
        choices=['silent', 'beep', 'voice'],
        help='Select sound feedback mode: silent/beep/voice'
    )

    parser.add_argument(
        '-v', '--visualize', action='store_true',
        help='visuzlize window')

    parser.add_argument(
        '-DG', '--DEBUG', action='store_true',
        help='DEBUG Mode')

    parser.add_argument(
        '--conf_th', type=float, default=0.3,
        help='confidence threshold')

    parser.add_argument(
        '-d', '--id', type=str, default='DRIVER001',
        help='device number')

    parser.add_argument(
        '--normal', type=int, default=25,  # [fix] 150 -> 25
        help='normal image save time (x seconds later save)')

    parser.add_argument(
        '--drowsiness', type=int, default=2,
        help='drowsiness detect time')

    parser.add_argument(
        '--phone', type=int, default=1,
        help='cell phone detect time')

    parser.add_argument(
        '--cigar', type=int, default=1,
        help='cigarette detect time')

    parser.add_argument(
        '--distraction', type=int, default=1,
        help='distraction detect time')

    parser.add_argument(
        '--yawn', type=int, default=2,
        help='yawn detect time')

    parser.add_argument(
        '--seconds_per_image', type=float, default=0.25,
        help='save image per seconds')

    parser.add_argument(
        '--max_image', type=int, default=20,
        help='max numer of save image each class')

    parser.add_argument(
        '--ear_t', type=float, default=0.2,
        help='ear threshold')

    parser.add_argument(
        '--lar_t', type=float, default=0.5,
        help='lar threshold')

    parser.add_argument(
        '--path_detector_plugin', type=str,
        default='plugins/libmyplugins_yolox.so'
    )
    parser.add_argument(
        '--path_detector_engine', type=str,
        # base yolox is medium version. also, you can use yolox_tiny.
        default='engines/yolox_m_icms_4cls.engine'
    )

    # parser.add_argument(
    #     '--path_face_landmark_estimator_plugin', type=str,
    #     default = 'plugins/libmyplugins_pflp.so'
    #     )
    parser.add_argument(
        '--path_face_landmark_estimator_engine', type=str,
        default='engines/resnet18.engine'
    )

    parser.add_argument(
        '--path_pipnet_engine', type=str,
        default='engines/pipnets.engine'
    )
    parser.add_argument('--meanface_text_file', type=str,
                        default='utils/PIPNET/meanface.txt')

    ##########################################################################

    parser.add_argument('--cellphone_use_log', type=str,
                        default='rec/CellphoneUse.txt')

    parser.add_argument('--cellphone_watch_log', type=str,
                        default='rec/CellphoneWatch.txt')

    parser.add_argument('--distraction_log', type=str,
                        default='rec/Distraction.txt')

    parser.add_argument('--drwosiness_log', type=str,
                        default='rec/Drwsiness.txt')

    parser.add_argument('--smoking_log', type=str,
                        default='rec/Smoking.txt')

    parser.add_argument('--history_log', type=str,
                        default='rec/History.txt')

##########################################################################

    args = parser.parse_args()
    return args


def parse_args_macos():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')

    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')

    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')

    parser.add_argument(
        '-ir', '--ignore_recording', action='store_true',
        help='Ignore Recording')

    parser.add_argument(
        '-s', '--sound_mode',
        default='silent',
        type=str,
        choices=['silent', 'beep', 'voice'],
        help='Select sound feedback mode: silent/beep/voice'
    )

    parser.add_argument(
        '-v', '--visualize', action='store_true',
        help='visuzlize window')

    parser.add_argument(
        '-DG', '--DEBUG', action='store_true',
        help='DEBUG Mode')

    parser.add_argument(
        '--conf_th', type=float, default=0.3,
        help='confidence threshold')

    parser.add_argument(
        '-d', '--id', type=str, default='DRIVER001',
        help='device number')

    parser.add_argument(
        '--normal', type=int, default=25,  # [fix] 150 -> 25
        help='normal image save time (x seconds later save)')

    parser.add_argument(
        '--drowsiness', type=int, default=2,
        help='drowsiness detect time')

    parser.add_argument(
        '--phone', type=int, default=1,
        help='cell phone detect time')

    parser.add_argument(
        '--cigar', type=int, default=1,
        help='cigarette detect time')

    parser.add_argument(
        '--distraction', type=int, default=1,
        help='distraction detect time')

    parser.add_argument(
        '--yawn', type=int, default=2,
        help='yawn detect time')

    parser.add_argument(
        '--seconds_per_image', type=float, default=0.25,
        help='save image per seconds')

    parser.add_argument(
        '--max_image', type=int, default=20,
        help='max numer of save image each class')

    parser.add_argument(
        '--ear_t', type=float, default=0.2,
        help='ear threshold')

    parser.add_argument(
        '--lar_t', type=float, default=0.5,
        help='lar threshold')

    parser.add_argument(
        '--path_yolov5_pt', type=str,
        default='weight/yolov5s_dms5cls.pt'
    )

    parser.add_argument(
        '--path_pflp', type=str,
        default='weight/pflp_160px_best_nme0.05_grayscale.pth'
    )

    parser.add_argument(
        '--img_width', type=int, default=1080,
        help='img_width')

    parser.add_argument(
        '--img_height', type=int, default=960,
        help='img_height')

    args = parser.parse_args()
    return args


def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--image', type=str, default=None,
                        help='image file name, e.g. dog.jpg')
    parser.add_argument('--video', type=str, default=None,
                        help='video file name, e.g. traffic.mp4')
    parser.add_argument('--video_looping', action='store_true',
                        help='loop around the video file [False]')
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')
    parser.add_argument('--usb', type=int, default=0,
                        help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--gstr', type=str, default=None,
                        help='GStreamer string [None]')
    parser.add_argument('--onboard', type=int, default=None,
                        help='Jetson onboard camera [None]')
    parser.add_argument('--copy_frame', action='store_true',
                        help=('copy video frame internally [False]'))
    parser.add_argument('--do_resize', action='store_true',
                        help=('resize image/video [False]'))
    parser.add_argument('--width', type=int, default=1280,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=960,
                        help='image height [480]')
    return parser