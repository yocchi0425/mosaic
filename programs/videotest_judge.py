""" A class for SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import pickle
import numpy as np
from random import shuffle
from scipy.misc.pilutil import imread,imresize
from timeit import default_timer as timer
import glob
import sys
sys.path.append("..")
from utils.ssd_utils import BBoxUtility
from tqdm import tqdm #プログレスバーを表示するモジュール


#モザイク処理をする関数
def mosaic(src, ratio=0.1):
        small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
      
def mosaic_area(src, xmin, ymin, xmax, ymax, ratio=0.05):
        dst = src.copy()
        return mosaic(dst[ymin:ymax, xmin:xmax], ratio)
    
def save(video_path):
        
        camera = cv2.VideoCapture(video_path) 

        # 動画ファイル保存用の設定
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter('video.mov', fourcc, fps, (w, h))

        # （qキーで撮影終了）
        while True:
            ret, frame = camera.read()                             # フレームを取得
            cv2.imshow('camera', frame)                            # フレームを画面に表示
            video.write(frame)                                     # 動画を1フレームずつ保存する
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()


class VideoTest(object):
    
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
        
        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col)
            

        
    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0):  
    	
    	#動画ファイル準備
        print('------------------------------------------------')
        print("input filename.mov or 0")
        print("input the name of video : ", end='')
        videoName = input()
        if videoName == '0':
            videoName = 'WebCam.mov'
            save(0)
            videoPass = 'video.mov'
            vid = cv2.VideoCapture(videoPass)
            total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)    
            tmp_key = 1
            print('Total frames : ', total_frames)
        else:
            videoPass = '../movies/' + videoName
            vid = cv2.VideoCapture(videoPass)
            total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)    
            tmp_key = 1
            print('Total frames : ', total_frames)
        print('------------------------------------------------')


        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        # Compute aspect ratio of video     
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vidar = vidw/vidh
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)
            
        #動画ファイル書き出し
        frame_rate=24
        fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
        f_v='../processed/' + videoName
        video=cv2.VideoWriter(f_v, fourcc, frame_rate, (int(vidw), int(vidh)))

        # プログレスバーを表示
        if tmp_key != 0:
            pbar = tqdm(total=total_frames)
        else:
            print('processing...')

        while vid.isOpened():
            #プログレスバーを進める
            if tmp_key != 0:
                pbar.update(1)

            # 全フレーム終了
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return
                
            im_size = (self.input_shape[0], self.input_shape[1])    
            resized = cv2.resize(orig_image, im_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Reshape to original aspect ratio for later visualization
            # The resized version is used, to visualize what kind of resolution
            # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1]))
            
            # Use model to predict 
            inputs = [image.img_to_array(rgb)]
            tmp_inp = np.array(inputs)
            x = preprocess_input(tmp_inp)
            
            y = self.model.predict(x)
            
            # This line creates a new TensorFlow device every time. Is there a 
            # way to avoid that?
            results = self.bbox_util.detection_out(y)
            
            if len(results) > 0 and len(results[0]) > 0:
                # Interpret output, only one frame is used 
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for i in range(top_conf.shape[0]):
                    class_num = int(top_label_indices[i])
                    # 30%以上と判定した場合
                    if (top_conf[i] > 0.3 and (self.class_names[class_num] == 'person')):
                        xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                        ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                        xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                        ymax = int(round(top_ymax[i] * to_draw.shape[0]))
                        
                        # 検出対象にモザイク処理をする
                        to_draw[ymin:ymax, xmin:xmax] = mosaic_area(to_draw, xmin, ymin, xmax, ymax)

            #検出のために拡大したサイズをもとに戻す
            to_draw = cv2.resize(to_draw,(int(vidw),int(vidh)))
            
            cv2.startWindowThread()
            #動画の表示
            cv2.imshow("SSD result", to_draw)
            #動画の書き出し
            video.write(to_draw)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        cv2.destroyAllWindows()
        print('finish')
        pbar.close()
        cap.release()
        video.release()