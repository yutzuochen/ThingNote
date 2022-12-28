import numpy as np
import requests
import json
import datetime
import time
import av
import os,sys,cv2,time,shutil
import tensorflow as tf
from tensorflow.python.platform import gfile
import keras
from PIL import Image,ImageDraw
import bbox_blur as bbox_util
import pylab as plt
from ffmpy import FFmpeg
from moviepy.editor import VideoFileClip
from matplotlib.pyplot import savefig
from nsfw import predict as nsfw_predict
from AIMakeup import makeup_ai as mkai
nsfw_model = nsfw_predict.load_model('./nsfw/nsfw.299x299.h5')
from dotenv import load_dotenv
import logging
load_dotenv()
logging.warning('logging level test')
from flask import Flask
from flask import request
from flask_cors import CORS
from flask_cors import cross_origin
app = Flask(__name__)
CORS(app)
def load_imgpil(imgpil,image_size):
	load_images=[]
	image=imgpil.resize(image_size)
	image=tf.keras.utils.img_to_array(image)
	image /= 255
	load_images.append(image)
	return np.asarray(load_images)
	
class harmony_protect():
	def __init__(self,pb_file_path='weights/inception_sp_0.9924_0.09_partialmodel.pb',mobile=False,USE_GPU=False):
		if not os.path.isfile(pb_file_path):
			print('file {} does not exist!'.format(pb_file_path))
			sys.exit()
		
		if USE_GPU==True:	
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
			config.gpu_options.allow_growth = True 
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
			config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
			
		g=tf.Graph()
		self.sess = tf.compat.v1.Session(graph=g,config=config) 
		with self.sess.as_default():
			with self.sess.graph.as_default():
				with gfile.FastGFile(pb_file_path, 'rb') as f:
					graph_def = tf.compat.v1.GraphDef()
					graph_def.ParseFromString(f.read())
					self.sess.graph.as_default()
					tf.import_graph_def(graph_def, name='')
				self.sess.run(tf.compat.v1.global_variables_initializer()) 
		
		
				self.input_img = self.sess.graph.get_tensor_by_name('input_1:0')    
				if mobile==False:
					self.conv_base_output=self.sess.graph.get_tensor_by_name('mixed10/concat:0')
					self.image_size=(299,299)
				else:
					self.conv_base_output=self.sess.graph.get_tensor_by_name('out_relu/Relu6:0')
					self.image_size=(224,224)
				print(self.input_img.shape,self.conv_base_output.shape)

		
	def classify(self, loaded_images,with_hp=False):
		if with_hp==False:
			raise('partial model does not support predict')
		else:
			heatmaps = self.sess.run(self.conv_base_output , feed_dict={self.input_img: loaded_images})
			if self.image_size==(299,299):
				heatmaps_avg=np.mean(heatmaps,axis=3).reshape((8,8)) 
			else:
				heatmaps_avg=np.mean(heatmaps,axis=3).reshape((7,7)) 
				
			return [],heatmaps_avg
	
	def classify_imgpil(self,imgpil,with_hp=False):
		w,h=imgpil.size
		loaded_image=load_imgpil(imgpil,self.image_size)
		if with_hp==True:
			ret,heatmaps=self.classify(loaded_image,True)
			heatmaps=cv2.resize(heatmaps,(w,h),interpolation=cv2.INTER_CUBIC)
			return ret,heatmaps
		ret,_=self.classify(loaded_image)
		return ret,_
		
	def general_harmony(self,imgpil,weight1=1.0,weight2=0.25):
		ret,hp=self.classify_imgpil(imgpil,True)
		bbox,heatmaps_index,max_value,avg_value=bbox_util.analyze_box(hp,weight1,weight2)
		imgblur=bbox_util.img_blur_2(imgpil,hp,heatmaps_index,max_value)
		return imgblur
	

def file_name(file_dir):   
	L=[] 
	for root, dirs, files in os.walk(file_dir):  
		for file in files:  
			if os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']:  
				L.append(os.path.join(root, file))  
	return L 

def call_zimg_api(pic_file, url):
    try:
        with open(pic_file, 'rb') as f:
            myobj = f.read()	
        x = requests.post(url, data = myobj)	
        res = x.json()
    except Exception as e:
        res = 'fail:'+ str(e)		
    return res

def get_url_pic(url, file_name):
    pic  = requests.get(url)
    img2 = pic.content
    pic_out = open(file_name,'wb')
    pic_out.write(img2) 
    pic_out.close() 

def nparraytofile(nparray, filename="your_file.jpg"):
    im = Image.fromarray(nparray)
    im.save(filename)


def filetonparray(filename):

    image = Image.open(filename)
    data = np.asarray(image)

    return data

def writeoutput_v2(res_file, fps, outfilename):
    images = []
    for i in res_file:
        images.append(filetonparray(i))
        os.remove(i)

    output = av.open(outfilename, 'w')
    stream = output.add_stream('h264', str(round(fps,2)))
    stream.bit_rate = 8000000

    for i, img in enumerate(images):
        if not i:
            stream.height = img.shape[0]
            stream.width = img.shape[1]        
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

    # flush
    packet = stream.encode(None)
    output.mux(packet)

    output.close()


def readFrame(video, file_, sample_fps=30):

    capture = cv2.VideoCapture(video)

    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    step = fps/ sample_fps
    cur_step = 0
    cur_count = 0
    save_count = 0
    res_frames = []
    while True:
        ret, frame = capture.read()
        if ret is False:
            break
        frame = np.array(frame)[:, :, :]
        cur_step += 1
        cur_count += 1
        if cur_step >= step:
            cur_step -= step
            save_count += 1
            res_frames.append(frame)
    capture.release()
    name_cnt = 0
    res_file = []
    for i in res_frames:
        filename_ = './'+ file_+ "_"+ str(name_cnt)+ ".jpg"
        nparraytofile(i, filename=filename_)
        res_file.append(filename_)
        name_cnt += 1

    return res_file, fps

def writeoutput(res_file, fps, outfilename):

    frames = []
    for i in res_file:
        frames.append(filetonparray(i))
        os.remove(i)
    fourcg = cv2.VideoWriter_fourcc(*'avc1')#cv2.VideoWriter_fourcc('H', '2', '6', '4')#cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#
    outfile = cv2.VideoWriter(outfilename, 
                            fourcg, 
                            fps, 
                            (int(frames[0].shape[1]), int(frames[0].shape[0])))

    for frame in frames:
        outfile.write(frame)
    
    outfile.release()
    cv2.destroyAllWindows()



def mosaic_pic_main(harmony, file_, nsfw_model):
    ex = ''
    try:
        nsfw_res = nsfw_predict.classify(nsfw_model, file_)[0]
        if nsfw_res['porn']+ nsfw_res['sexy'] >= 0.55 or nsfw_res['hentai'] >= 0.6:
            img=Image.open(file_).convert('RGB')
            ret,hp=harmony.classify_imgpil(img,True)
            imgblur_general=harmony.general_harmony(img,weight1=0.35,weight2=0.2)
            w,h=img.size
            ratio=w/h
            plt.figure(figsize=(w/96,h/96))
            plt.imshow(imgblur_general)
            plt.title('general',fontsize=8)
            plt.axis('off')	
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top =1, bottom = 0, right =1, left = 0, hspace = 0.05, wspace = 0.05)
            plt.margins(0, 0)
            plt.savefig(file_,dpi=160)
            plt.close()
        res = True
        ex = ex
    except Exception as e:
        file_ = file_
        res = False
        ex += str(e)
    return file_, res, ex


def waterprint_main(file_):
    ex = ''
    try:
        watermark_jpg = 'watermark.png'
        img = cv2.imread(file_)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 將圖片轉成灰階
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   # 載入人臉模型
        faces = face_cascade.detectMultiScale(gray, 1.2,3)    # 偵測人臉

        img = Image.open(file_)
        icon = Image.open(watermark_jpg)
        img_w, img_h = img.size
        icon_w, icon_h = icon.size

        label_rightdown = (int(img_w-icon_w), int(img_h-icon_h), icon_w, icon_h)
        label_rightup = (int(img_w-icon_w), int(0), icon_w, icon_h)
        label_leftdown = (int(0), int(img_h-icon_h), icon_w, icon_h)
        label_leftup = (int(0), int(0), icon_w, icon_h)
        chosen = [label_leftdown, label_rightdown, label_rightup, label_leftup]
        if len(faces) == 0:
            x = label_rightdown[0]
            y = label_rightdown[1]
        else:
            res = []
            for watermark in chosen:
                res_ = 0
                for face in faces:
                    rec_du = (face[0], face[1], face[2], face[3])  
                    rec_s = watermark
                    res_ += IOU(rec_du, rec_s)
                res.append(res_/len(faces))
            final_xy = chosen[get_minvalue(res)]
            x = final_xy[0]
            y = final_xy[1]
        img.paste(icon, (x, y), icon)   
        img.save(file_)
        ret = True
        ex = ex
    except Exception as e:
        file_ = file_
        ret = False
        ex += str(e)
    return file_, ret, ex

def waterprint_video_main(file_, nb):
    ex = ''
    try:
        watermark_jpg = 'watermark.png'

        img = Image.open(file_)
        img2 = Image.open(file_) 

        icon = Image.open(watermark_jpg)
        img_w, img_h = img.size
        icon_w, icon_h = icon.size

        label_leftdown = (int(0), int(img_h-icon_h), icon_w, icon_h)
        x = label_leftdown[0]
        y = label_leftdown[1]
        img.paste(icon, (x, y), icon)   
        img.convert('RGBA')
        img.putalpha(nb)
        img2.paste(img,(0,0),img)  
        img2.save(file_)
        ret = True
        ex = ex
    except Exception as e:
        file_ = file_
        ret = False
        ex += str(e)
    return file_, ret, ex


def video_add_audio(video_path: str, audio_path: str, output_dir: str, outputfile_name:str):
    _ext_video = os.path.basename(video_path).strip().split('.')[-1]
    _ext_audio = os.path.basename(audio_path).strip().split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']:
        raise Exception('audio format not support')
    _codec = 'copy'
    if _ext_audio == 'wav':
        _codec = 'aac'
    result = os.path.join(output_dir, outputfile_name)
    ff = FFmpeg(
        executable='ffmpeg',
        inputs={video_path: None, audio_path: None},
        outputs={result: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    #print(ff.cmd)
    ff.run()
    #ff.process.terminate()
    return result

def IOU(Reframe, GTframe):
    # step1：get x,y,w,h from ta1 matrix
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    # step2：get x,y,w,h from ta2 matrix
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]
    # 計算重疊部分的寬跟高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重疊部分為負, 即不重疊
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1.0 / (Area1 + Area2 - Area)

    return ratio

def get_minvalue(inputlist):
    min_value = min(inputlist)
    min_index = inputlist.index(min_value)
    return min_index




@app.route('/AIMakeUp/')
def AImakeup_api():

    zimg_domain = os.getenv('zimg_domain')
    url = zimg_domain+ "upload"
	#step1：接收傳的圖片
    ##parameter
    pic_file = request.args.get('file')#parameter1
    value_white = float(str(request.args.get('value_white'))) * 0.7#美白
    value_smooth = float(str(request.args.get('value_smooth')))#磨皮
    value_mouth_brightening = float(str(request.args.get('value_mouth_brightening')))#紅唇
    value_sharp = float(str(request.args.get('value_sharp')))#亮眼
    
    #
    url_post = zimg_domain+ str(pic_file)
    #file
    file_name = str(pic_file)
    file_ = f"./{file_name}.jpg"
    if url_post == '':
        res_final = {"status":0,
                     'msg':'no url',
                     'result': 'no url'}
    else:
        try:
            get_url_pic(url_post, file_name=file_)
            mu = mkai.Makeup('./AIMakeup/data/shape_predictor_68_face_landmarks.dat')
            im,temp_bgr,faces=mu.read_and_mark(file_)
            imc=im.copy()
            for face in faces[file_]:
                face.whitening(value_white)
                face.smooth(0.7)
                face.organs['forehead'].whitening(value_white)
                face.organs['forehead'].smooth(0.7)
                face.organs['mouth'].brightening(value_mouth_brightening)
                face.organs['mouth'].smooth(value_smooth)
                face.organs['mouth'].whitening(value_white)
                face.organs['left eye'].whitening(value_white)
                face.organs['right eye'].whitening(value_white)
                face.organs['left eye'].sharpen(value_sharp)
                face.organs['right eye'].sharpen(value_sharp)
                face.organs['left eye'].smooth(value_smooth)
                face.organs['right eye'].smooth(value_smooth)
                face.organs['left brow'].whitening(value_white)
                face.organs['right brow'].whitening(value_white)
                face.organs['left brow'].sharpen(value_sharp)
                face.organs['right brow'].sharpen(value_sharp)
                face.organs['nose'].whitening(value_white)
                face.organs['nose'].smooth(0.7)
                face.organs['nose'].sharpen(value_sharp)
                face.sharpen(value_sharp)
            cv2.imwrite(file_, im)
            res_final = call_zimg_api(file_, url)
        except Exception as e:
            res_final = {"status":0,
                        'msg': str(e),
                        'result': pic_file}	
    if os.path.exists(file_):
        os.remove(file_)
    ret = json.dumps(res_final, indent=4, ensure_ascii=False)
    return ret

@app.route('/addwaterprint/')
def route_addwaterprint():	
	#execute
    zimg_domain = os.getenv('zimg_domain')
    url = zimg_domain+ "upload"
	#step1：接收傳的圖片
    pic_file = request.args.get('file')
    url_post = zimg_domain+ str(pic_file)
    #file
    file_name = str(pic_file)
    file_ = f"./{file_name}.png"
    if url_post == '':
        res_final = {"status":0,
                     'msg':'no url',
                     'result': 'no url'}
    else:
        try:
            get_url_pic(url_post, file_name=file_)
            file_, res, exce = waterprint_main(file_)
            res_final = call_zimg_api(file_, url)

        except Exception as e:
            res_final = {"status":0,
                         'msg': str(e),
                         'result': pic_file}	
    ret = json.dumps(res_final, indent=4, ensure_ascii=False)
    return ret


@app.route('/waterprint/')
def route_waterprint():	
	#execute
    zimg_domain = os.getenv('zimg_domain')
    url = zimg_domain+ "upload"
	#step1：接收傳的圖片/影片
    pic_file = request.args.get('file')
    pic_file = str(pic_file)    
    format_type = request.args.get('type')#['pic', 'video']
    format_type = str(format_type)

    #
    url_post = zimg_domain+ pic_file
    #step1：接收傳的圖片/影片
    if url_post == '':
        res_final = {"status":0,
                    'msg': 'no pic/video url',
                    'result': ''}	
    else:
        try:
            if format_type == 'pic':
                file_ = f"./{pic_file}.png"
                file_song = f"./{pic_file}_song.wav"
                file_final = f"./{pic_file}_res.png"
                file_final_addaudio = f"./{pic_file}_res_addaudio.mp4"
                get_url_pic(url_post, file_name=file_)

                file_, res, e = waterprint_main(file_)
                if e == '':
                    res_final = call_zimg_api(file_, url)
                else:
                    res_final = {"status":0,
                                 "msg": str(e),
                                 "result": pic_file}	

            elif format_type == 'video':
                file_ = f"./{pic_file}.mp4"
                file_song = f"./{pic_file}_song.wav"
                file_final = f"./{pic_file}_res.mp4"
                file_final_addaudio = f"./{pic_file}_res_addaudio.mp4"
                get_url_pic(url_post, file_name=file_)
                #
                video = VideoFileClip(file_)   # 讀取影片
                audio = video.audio # 取出聲音
                res_file, fps_ = readFrame(file_, pic_file, sample_fps=30)
                file_all = []
                for i in res_file:
                    #file_tmp, res, exce = waterprint_main(i)
                    file_tmp, res, exce = waterprint_video_main(i,125)
                    file_all.append(file_tmp)
                writeoutput_v2(res_file, fps_, file_final)
                if audio is not None:
                    audio.write_audiofile(file_song)         #file_song 輸出聲音為 wav 
                    video_add_audio(file_final, file_song, os.getcwd(), file_final_addaudio)
                    res_final = call_zimg_api(file_final_addaudio, url)
                elif audio is  None:
                    res_final = call_zimg_api(file_final, url)
        except Exception as ex:
            res_final = {"status":0,
                         "msg": str(ex),
                         "result": pic_file}
        if os.path.exists(file_):
            os.remove(file_)
        if os.path.exists(file_final):
            os.remove(file_final)
        if os.path.exists(file_song):
            os.remove(file_song)    
        if os.path.exists(file_final_addaudio):
            os.remove(file_final_addaudio)                                                                           
    ret = json.dumps(res_final, indent=4, ensure_ascii=False)
    return ret

# if __name__ == "__main__":
#     try:
#         app.run(host='0.0.0.0')
#     except Exception as e:
#         logging.warning('error', exc_info=True)
# @app.route('/mosaic/')
# def route_mosaic():
def pictureProcess():
	#execute
    harmony = harmony_protect()
	#step1：接收傳的圖片/影片
    # pic_file = request.args.get('file')
    # pic_file = str(pic_file)    
    # format_type = request.args.get('type')#['pic', 'video']
    # format_type = str(format_type)
    # url_post = zimg_domain+ pic_file
    
    # mason:can lonly input png?
    # file_ = f"/root/Desktop/first/masonMosaic/sirsir_1.jpg" #mason:???
    # get_url_pic(url_post, file_name=file_)
    # pic_out = open(file_name,'wb')
    # pic_out.write(img2) 
    # pic_out.close() 

    # file_, res, e = mosaic_pic_main(harmony, file_, nsfw_model)
    # if e == '':
    #     res_final = call_zimg_api(file_, url)
    # else:
    #     res_final = {"status":0,
    #                 "msg": str(e),
    #                 "result": pic_file}	
          
    # ret = json.dumps(res_final, indent=4, ensure_ascii=False)
    return res

pictureProcess()