IS_DLIB_INSTALLED = False
try:
    import dlib
    IS_DLIB_INSTALLED = True
except ImportError:
    pass

import torch
#import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
import comfy.utils
import comfy.model_management as mm
import gc
from tqdm import tqdm
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import folder_paths
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
script_directory = os.path.dirname(os.path.abspath(__file__))

try:
    from .liveportrait.utils.cropper import CropperFaceAlignment
except:
    log.warning("Can't load FaceAlignment, CropperFaceAlignment not available")
try:
    from .liveportrait.utils.cropper import CropperMediaPipe
except:
    log.warning("Can't load MediaPipe, MediaPipeCropper not available")
try:
    from .liveportrait.utils.cropper import CropperInsightFace
except:
    log.warning("Can't load MediaPipe, MediaPipeCropper not available")

from .liveportrait.utils.crop import _transform_img_kornia

DLIB_DIR = os.path.join(folder_paths.models_dir, "dlib")
class DLib:
    def __init__(self, predictor=68):
        self.face_detector = dlib.get_frontal_face_detector()
        # check if the models are available
        if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat")):
            raise Exception("The 5 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_5_face_landmarks.dat")
        if not os.path.exists(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")):
            raise Exception("The face recognition model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/dlib_face_recognition_resnet_model_v1.dat")
        self.predictor=predictor
        if predictor == 81:
            self.shape_predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_81_face_landmarks.dat"))
        elif predictor == 5:
            self.shape_predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat"))
        else:
            self.shape_predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat"))

        self.face_recognition = dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat"))
        #self.thresholds = THRESHOLDS["Dlib"]

    def get_face(self, image):
        faces = self.face_detector(np.array(image), 1)
        #faces, scores, _ = self.face_detector.run(np.array(image), 1, -1)
        
        if len(faces) > 0:
            return sorted(faces, key=lambda x: x.area(), reverse=True)
            #return [face for _, face in sorted(zip(scores, faces), key=lambda x: x[0], reverse=True)] # sort by score
        return None
            # 检测面部并提取关键点
    def get_landmarks(self, image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray)
                if len(faces) == 0:
                    return None
                shape = self.shape_predictor(gray, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                if self.predictor == 81:
                    landmarks = np.concatenate((landmarks[:17], landmarks[68:81]))
                    return landmarks
                elif self.predictor == 5:
                    return landmarks
                else:
                    return landmarks[:17]

    def get_all_landmarks(self, image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray)
                if len(faces) == 0:
                    return None
                shape = self.shape_predictor(gray, faces[0])
                output = np.array([[p.x, p.y] for p in shape.parts()])
                if self.predictor == 81:
                    leftEye=np.mean( output[36:42],axis=0)
                    rightEye=np.mean( output[42:48],axis=0)
                    mouth=np.mean( output[48:68],axis=0)
                elif self.predictor == 5:
                    leftEye=np.mean( output[0:3],axis=0)
                    rightEye=np.mean( output[2:4],axis=0)
                    mouth=output[4]
                else:
                    leftEye=np.mean( output[36:42],axis=0)
                    rightEye=np.mean( output[42:48],axis=0)
                    mouth=np.mean( output[48:68],axis=0)

                return output,leftEye,rightEye,mouth
                
    def draw_landmarks(self, image, landmarks, color=(255, 0, 0), radius=3):
            # cv2.circle打坐标点的坐标系，如下。左上角是原点，先写x再写y
            #  (0,0)-------------(w,0)
            #  |                  |
            #  |                  |
            #  (0,h)-------------(w,h)|
                #font = cv2.FONT_HERSHEY_SIMPLEX
                image_cpy = image.copy()
                for n in range(landmarks.shape[0]):
                    try:
                        cv2.circle(image_cpy, (int(landmarks[n][0]), int(landmarks[n][1])), radius, color, -1)
                    except:
                         pass
                    #cv2.putText(image_cpy, str(n), (landmarks[n][1], landmarks[n][0]), font, 0.5, color, 1, cv2.LINE_AA)
                return image_cpy
    
    def interpolate(self, image1, image2,landmarkType,AlignType,GenLandMarkImg):

            height,width = image1.shape[:2]
            w=width
            h=height

            if landmarkType == "ALL" or AlignType == "Landmarks":
                landmarks1,leftEye1,rightEye1,mouth1 = self.get_all_landmarks(image1)
                landmarks2,leftEye2,rightEye2,mouth2 = self.get_all_landmarks(image2)
                
            else:
                landmarks1 = self.get_landmarks(image1)
                landmarks2 = self.get_landmarks(image2)                 

            #画面划分成16*16个区域，然后去掉边界框以外的区域。
            src_points = np.array([
                [x, y]
                for x in np.linspace(0, w, 16)
                for y in np.linspace(0, h, 16)
            ])
            
            #上面这些区域同时被加入src和dst，使这些区域不被拉伸（效果是图片边缘不被拉伸）
            src_points = src_points[(src_points[:, 0] <= w/8) | (src_points[:, 0] >= 7*w/8) |  (src_points[:, 1] >= 7*h/8)| (src_points[:, 1] <= h/8)]
            #mark_img = self.draw_landmarks(mark_img, src_points, color=(255, 0, 255))
            dst_points = src_points.copy()


            #不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标
            dst_points = np.append(dst_points,landmarks1,axis=0)

            #变形目标人物的landmarks，先计算边界框
            landmarks2=np.array(landmarks2)
            min_x = np.min(landmarks2[:, 0])
            max_x = np.max(landmarks2[:, 0])
            min_y = np.min(landmarks2[:, 1])
            max_y = np.max(landmarks2[:, 1])
            #得到目标人物的边界框的长宽比
            ratio2 = (max_x - min_x) / (max_y - min_y)

            #变形原始人物的landmarks，边界框
            landmarks1=np.array(landmarks1)
            min_x = np.min(landmarks1[:, 0])
            max_x = np.max(landmarks1[:, 0])
            min_y = np.min(landmarks1[:, 1])
            max_y = np.max(landmarks1[:, 1])
            #得到原始人物的边界框的长宽比以及中心点
            ratio1 = (max_x - min_x) / (max_y - min_y)
            middlePoint = [ (max_x + min_x) / 2, (max_y + min_y) / 2]

            landmarks1_cpy = landmarks1.copy()

            if AlignType=="Width":
            #保持人物脸部边界框中心点不变，垂直方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致
                landmarks1_cpy[:, 1] = (landmarks1_cpy[:, 1] - middlePoint[1]) * ratio1 / ratio2 + middlePoint[1]
            elif AlignType=="Height":
            #保持人物脸部边界框中心点不变，水平方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致
                landmarks1_cpy[:, 0] = (landmarks1_cpy[:, 0] - middlePoint[0]) * ratio2 / ratio1 + middlePoint[0]
            elif AlignType=="Landmarks":
                MiddleOfEyes1 = (leftEye1+rightEye1)/2
                MiddleOfEyes2 = (leftEye2+rightEye2)/2

                # angle = float(np.degrees(np.arctan2(leftEye2[1] - rightEye2[1], leftEye2[0] - rightEye2[0])))
                # angle -= float(np.degrees(np.arctan2(leftEye1[1] - rightEye1[1], leftEye1[0] - rightEye1[0])))
                # rotation_matrix = np.array([
                #     [np.cos(angle), -np.sin(angle)],
                #     [np.sin(angle), np.cos(angle)]
                # ])

                distance1 =  ((leftEye1[0] - rightEye1[0]) ** 2 + (leftEye1[1] - rightEye1[1]) ** 2) ** 0.5
                distance2 =  ((leftEye2[0] - rightEye2[0]) ** 2 + (leftEye2[1] - rightEye2[1]) ** 2) ** 0.5
                factor = distance1 / distance2
                # print("distance1:",distance1)
                # print("distance2:",distance2)
                # print("factor:",factor)
                # print("MiddleOfEyes1:",MiddleOfEyes1)
                # print("MiddleOfEyes2:",MiddleOfEyes2)
                # print("angle:",angle)
                MiddleOfEyes2 = np.array(MiddleOfEyes2)
                
                landmarks1_cpy = (landmarks2 - MiddleOfEyes2) * factor + MiddleOfEyes1
                
                #landmarks1_cpy = landmarks1_cpy + MiddleOfEyes1

                            # landmarks1_cpy = (landmarks2 - MiddleOfEyes2) * factor
                            # landmarks1_cpy = landmarks1_cpy.T

                            # # 旋转坐标
                            # rotated_landmarks = np.dot(rotation_matrix, landmarks1_cpy)

                            # # 将旋转后的坐标转换回行向量
                            # rotated_landmarks = rotated_landmarks.T
                            # # 将 MiddleOfEyes1 转换为二维数组
                            # MiddleOfEyes1 = np.array(MiddleOfEyes1)

                            # # 将 landmarks1_cpy 和 MiddleOfEyes1_expanded 相加
                            # landmarks1_cpy = landmarks1_cpy + MiddleOfEyes1


            #不知道原作者为何把这个数组叫src，其实这是变形后的坐标
            src_points = np.append(src_points,landmarks1_cpy,axis=0)
            #print(landmarks1_cpy)
            
            mark_img = self.draw_landmarks(image1, dst_points, color=(255, 255, 0),radius=4)
            mark_img = self.draw_landmarks(mark_img, src_points, color=(255, 0, 0),radius=3)
            
            # Create the RBF interpolator instance            
            #Tried many times, finally find out these array should be exchange w,h before go into RBFInterpolator            
            src_points[:, [0, 1]] = src_points[:, [1, 0]]
            dst_points[:, [0, 1]] = dst_points[:, [1, 0]]

            rbfy = RBFInterpolator(src_points,dst_points[:,1],kernel="thin_plate_spline")
            rbfx = RBFInterpolator(src_points,dst_points[:,0],kernel="thin_plate_spline")

            # Create a meshgrid to interpolate over the entire image
            img_grid = np.mgrid[0:height, 0:width]

            # flatten grid so it could be feed into interpolation
            flatten=img_grid.reshape(2, -1).T

            # Interpolate the displacement using the RBF interpolators
            map_y = rbfy(flatten).reshape(height,width).astype(np.float32)
            map_x = rbfx(flatten).reshape(height,width).astype(np.float32)
            # Apply the remapping to the image using OpenCV
            warped_image = cv2.remap(image1, map_y, map_x, cv2.INTER_LINEAR)

            if GenLandMarkImg:
                return warped_image, mark_img
            else:
                return warped_image, warped_image
       
class FaceShaperModels:
    @classmethod
    def INPUT_TYPES(s):
        libraries = []
        if IS_DLIB_INSTALLED:
            libraries.append("dlib")

        return {"required": {
            "DetectType": ([81,68,5], ),
        }}

    RETURN_TYPES = ("FaceShaper_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceShaper"

    def load_models(self, DetectType):
        out = {}

        # if library == "insightface":
        #     out = InsightFace(provider)
        # else:
        #     out = DLib()
        out = DLib(DetectType)
        return (out, )



def draw_pointsOnImg(image, landmarks, color=(255, 0, 0), radius=3):
        # cv2.circle打坐标点的坐标系，如下。左上角是原点，先写x再写y
        #  (0,0)-------------(w,0)
        #  |                  |
        #  |                  |
        #  (0,h)-------------(w,h)|
    image_cpy = image.copy()
    for n in range(landmarks.shape[0]):
        try:
            cv2.circle(image_cpy, (int(landmarks[n][0]), int(landmarks[n][1])), radius, color, -1)                        
        except:
                pass
    return image_cpy

def write_pointsOnImg(image, landmarks, color=(255, 0, 0), fontsize=0.25):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_cpy = image.copy()
    for n in range(landmarks.shape[0]):
        try:
            cv2.putText(image_cpy, str(n), (int(landmarks[n][0]), int(landmarks[n][1])), font, fontsize, color, 1, cv2.LINE_AA)
            cv2.circle(image_cpy, (int(landmarks[n][0]), int(landmarks[n][1])), 1, color, -1) 
        except:
                pass
    return image_cpy

def drawLineBetweenPoints(image, pointsA, pointsB, color=(255, 0, 0), thickness=1):
    image_cpy = image.copy()
    for n in range(pointsA.shape[0]):
        try:
            cv2.line(image_cpy, (int(pointsA[n][0]), int(pointsA[n][1])), (int(pointsB[n][0]), int(pointsB[n][1])), color, thickness)                        
        except:
            pass
    return image_cpy

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    #return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)

def mask_from_landmarks(height, width, landmarks):
    #import cv2
    mask = np.zeros((height, width), dtype=np.float64)
    points = cv2.convexHull(landmarks)
    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, color=1)
    return mask

def expand_mask(mask, expand, tapered_corners):
    import scipy

    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0)

class FaceShaper:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("FaceShaper_MODELS", ),
                "imageFrom": ("IMAGE",),
                "imageTo": ("IMAGE",),
                "landmarkType": (["ALL","OUTLINE"], ),
                "AlignType":(["Width","Height","Landmarks"], ),
                #"TargetFlip":([True,False],),
            },
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("Image1","LandmarkImg")
    FUNCTION = "run"

    CATEGORY = "FaceShaper"

    def run(self,analysis_models,imageFrom, imageTo,landmarkType,AlignType):
        tensor1 = imageFrom*255
        tensor1 = np.array(tensor1, dtype=np.uint8)
        tensor2 = imageTo*255
        tensor2 = np.array(tensor2, dtype=np.uint8)
        output=[]
        image1 = tensor1[0]
        image2 = tensor2[0]
        
        img1,img2 = analysis_models.interpolate(image1,image2,landmarkType,AlignType,True)
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).unsqueeze(0)               
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0).unsqueeze(0)  
        output.append(img1)
        output.append(img2)
 
        return (output)
class FaceShaperShowLandMarks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "crop_info": ("CROPINFO", {"default": []}),
            "landmark_lines": ("BOOLEAN", {"default": False}),
            "writeIndexOnKeyPoints": ("BOOLEAN", {"default": False}),            
            "fontSize": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 5, "step": 0.05}),
            "rescaleCroppedImg": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
            },
            
            "optional": {
                "croppedImg": ("IMAGE", {"default": None}),
                "sourceImg": ("IMAGE", {"default": None}),
                
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("landmark on source","landmark on cropped","keyPoints on cropped")
    FUNCTION = "run"
    CATEGORY = "FaceShaper"

    def run(self, crop_info, landmark_lines, writeIndexOnKeyPoints,fontSize,rescaleCroppedImg,croppedImg=None, sourceImg=None):
        if sourceImg is not None:
            sourceImg = sourceImg*255
            sourceImg = np.array(sourceImg, dtype=np.uint8)
        if croppedImg is not None:
            croppedImg = croppedImg*255
            croppedImg = np.array(croppedImg, dtype=np.uint8)
        #output=[]
        lmk = crop_info["crop_info_list"][0]['lmk_source']
        height, width = crop_info["crop_info_list"][0]['input_image_size']
        img1=self.draw203keypoints(lmk, landmark_lines, sourceImg,height, width,writeIndexOnKeyPoints)
        #output.append(img1)
        lmk_crop = crop_info["crop_info_list"][0]['lmk_crop']

        dsize = crop_info["crop_info_list"][0]['dsize']
        img2=self.draw203keypoints(lmk_crop, landmark_lines, croppedImg,dsize,dsize,writeIndexOnKeyPoints)
        #output.append(img2)
        if writeIndexOnKeyPoints:
            img3 = (self.writePt_crop(crop_info,croppedImg, fontSize,rescaleCroppedImg))
            #output.append(img3)
        else:
            img3=(self.drawPt_crop(crop_info,croppedImg,rescaleCroppedImg))

        #print("img3",img3.shape) #it is torch.Size([1, 1024, 1024, 3])
        output=[]
        output.append(img1)
        output.append(img2)
        output.append(img3)
        return (output)
        

    def draw203keypoints(self, lmk, draw_lines, sourceImg,height,width,writeIndex):
        #           left upper eye | left lower eye | right upper eye | right lower eye | upper lip top | lower lip bottom | upper lip bottom | lower lip top | jawline         | left eyebrow | right eyebrow | nose            | left pupil    | right pupil  |  nose center
        indices = [                  12,               24,              37,               48,             66,                85,                96,             108,              145,           165,            185,             197,             198,            199,          203]
        colorlut = [(0, 0, 255),     (0, 255, 0),     (0, 0, 255),      (0, 255, 0),      (255, 0, 0),    (255, 0, 255),     (255, 255, 0),     (0, 255, 255),  (128, 128, 128), (128, 128, 0), (128, 128, 0),   (0,128,128),    (255, 255,255),   (255, 255,255), (255,255,255)]
        colors = []
        c = 0
        for i in range(203):
            if i == indices[c]:
                c+=1
            colors.append(colorlut[c])
        if sourceImg is not None:
            target_image = sourceImg[0].copy()
        else:
            target_image = np.zeros((height, width, 3), dtype=np.uint8) * 255

        keypoints_img_list = []
        keypoints = lmk.copy()        
        if draw_lines:
            start_idx = 0
            for end_idx in indices:
                color = colors[start_idx]
                for i in range(start_idx, end_idx - 1):
                    pt1 = tuple(map(int, keypoints[i]))
                    pt2 = tuple(map(int, keypoints[i+1]))
                    if all(0 <= c < d for c, d in zip(pt1 + pt2, (width, height) * 2)):
                        cv2.line(target_image, pt1, pt2, color, thickness=1)
                if end_idx == start_idx +1:
                    x,y = keypoints[start_idx]
                    cv2.circle(target_image, (int(x), int(y)), radius=1, thickness=-1, color=colors[start_idx])
                        
                start_idx = end_idx
        else:
            for index, (x, y) in enumerate(keypoints):
                cv2.circle(target_image, (int(x), int(y)), radius=1, thickness=-1, color=colors[index])
                if(writeIndex):
                    cv2.putText(target_image, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        #keypoints_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(target_image.astype(np.float32) / 255.0).unsqueeze(0) 
        keypoints_img_list.append(keypoints_image)
        keypoints_img_tensor = (
            torch.stack([torch.from_numpy(np_array) for np_array in keypoints_img_list]) / 255).float()

        return (keypoints_img_tensor,)
    
    def drawPt_crop(self, crop_info, croppedImg,ImageSize):
        pbar = comfy.utils.ProgressBar(len(crop_info))
        c=0
        output=[]
        for crop in crop_info["crop_info_list"]:
            if crop:
                if ImageSize != 512:
                    inputImg = cv2.resize(croppedImg[c], (ImageSize, ImageSize), interpolation=cv2.INTER_AREA)
                    keys = crop['pt_crop'].copy()
                    keys = keys * (ImageSize / 512)
                    mark_img = draw_pointsOnImg(inputImg, keys, color=(255, 0, 0),radius=3)
                else:
                    mark_img = draw_pointsOnImg(croppedImg[c], crop['pt_crop'], color=(255, 0, 0),radius=3)
            else:
                mark_img = croppedImg[c].copy()
            c += 1
            mark_img= torch.from_numpy(mark_img.astype(np.float32) / 255.0).unsqueeze(0)  
            return mark_img
            output.append(mark_img)
            pbar.update(1)
        return torch.stack(output,)
#cropped_image_256 = cv2.resize(image_crop, (256, 256), interpolation=cv2.INTER_AREA)
    def writePt_crop(self, crop_info, croppedImg, fontSize,ImageSize):
        pbar = comfy.utils.ProgressBar(len(crop_info))
        c=0
        output=[]
        for crop in crop_info["crop_info_list"]:
            if crop:
                if ImageSize != 512:
                    inputImg = cv2.resize(croppedImg[c], (ImageSize, ImageSize), interpolation=cv2.INTER_AREA)
                    keys = crop['lmk_crop'].copy()
                    keys = keys * (ImageSize / 512)
                    mark_img = write_pointsOnImg(inputImg, keys, color=(255, 0, 0),fontsize=fontSize)
                else:
                    mark_img = write_pointsOnImg(croppedImg[c].copy(), crop['lmk_crop'], color=(255, 0, 0),fontsize=fontSize)
            else:
                mark_img = croppedImg[c].copy()
            c += 1
            mark_img= torch.from_numpy(mark_img.astype(np.float32) / 255.0).unsqueeze(0)  
            return mark_img
            output.append(mark_img)
            pbar.update(1)
        return torch.stack(output,)

class FaceShaperLoadInsightFaceCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "onnx_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML'], {
                        "default": 'CPU'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FSMCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "FaceShaper"

    def crop(self, onnx_device, keep_model_loaded, detection_threshold=0.5):
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': onnx_device,
            'detection_threshold': detection_threshold
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperInsightFace(**cropper_init_config)

        return (self.cropper,)

class FaceShaperLoadMediaPipeCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "landmarkrunner_onnx_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML', 'torch_gpu'], {
                        "default": 'CPU'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": True})
            },           
        }

    RETURN_TYPES = ("FSMCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "FaceShaper"

    def crop(self, landmarkrunner_onnx_device, keep_model_loaded):
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': landmarkrunner_onnx_device
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperMediaPipe(**cropper_init_config)

        return (self.cropper,)
        
class FaceAlignmentCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "face_detector": (
                    ['blazeface', 'blazeface_back_camera', 'sfd'], {
                        "default": 'blazeface_back_camera'
                    }),

            "landmarkrunner_device": (
                    ['CPU', 'CUDA', 'ROCM', 'CoreML', 'torch_gpu'], {
                        "default": 'torch_gpu'
                    }),
            "face_detector_device": (
                    ['cuda', 'cpu', 'mps'], {
                        "default": 'cuda'
                    }),

            "face_detector_dtype": (
                    [
                        "fp16",
                        "bf16",
                        "fp32",
                    ],
                    {"default": "fp16"},
                ),
            "keep_model_loaded": ("BOOLEAN", {"default": True})

            },           
        }

    RETURN_TYPES = ("FSMCROPPER",)
    RETURN_NAMES = ("cropper",)
    FUNCTION = "crop"
    CATEGORY = "FaceShaper"

    def crop(self, landmarkrunner_device, keep_model_loaded, face_detector, face_detector_device, face_detector_dtype):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[face_detector_dtype]
        cropper_init_config = {
            'keep_model_loaded': keep_model_loaded,
            'onnx_device': landmarkrunner_device,
            'face_detector_device': face_detector_device,
            'face_detector': face_detector,
            'face_detector_dtype': dtype
        }
        
        if not hasattr(self, 'cropper') or self.cropper is None or self.current_config != cropper_init_config:
            self.current_config = cropper_init_config
            self.cropper = CropperFaceAlignment(**cropper_init_config)

        return (self.cropper,)
    
class FaceShaperCropper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cropper": ("FSMCROPPER",),
            "source_image": ("IMAGE",),
            "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
            "scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 4.0, "step": 0.01}),
            "vx_ratio": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            "vy_ratio": ("FLOAT", {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.001}),
            "face_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            "face_index_order": (
                    [
                        'large-small', 
                        'left-right', 
                        'right-left',
                        'top-bottom',
                        'bottom-top',
                        'small-large',
                        'distance-from-retarget-face'
                     ],
                    ),
            "rotate": ("BOOLEAN", {"default": True}),
            },           
        }

    RETURN_TYPES = ("IMAGE", "CROPINFO",)
    RETURN_NAMES = ("cropped_image", "crop_info",)
    FUNCTION = "process"
    CATEGORY = "FaceShaper"

    def process(self, cropper, source_image, dsize,scale, vx_ratio, vy_ratio, face_index, face_index_order, rotate):
        source_image_np = (source_image.contiguous() * 255).byte().numpy()

        # Initialize lists
        crop_info_list = []
        cropped_images_list = []
        # source_info = []
        # source_rot_list = []
        # f_s_list = []
        # x_s_list = []

        #KJ原版代码里，这个值是输入参数，取值范围64-2048，默认512
        #会影响后面的composite操作。本人来不及修改后面的composite操作，所以这里直接写死512
        #dsize = 512
        # Initialize a progress bar for the combined operation
        pbar = comfy.utils.ProgressBar(len(source_image_np))
        for i in tqdm(range(len(source_image_np)), desc='Detecting, cropping, and processing..', total=len(source_image_np)):
            # Cropping operation
            crop_info, cropped_image = cropper.crop_single_image(source_image_np[i], dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate)
            
            # Processing source images
            if crop_info:
                crop_info['dsize'] = dsize
                crop_info_list.append(crop_info)

                cropped_images_list.append(cropped_image)

                # f_s_list.append(None)
                # x_s_list.append(None)
                # source_info.append(None)
                # source_rot_list.append(None)
                
            else:
                log.warning(f"Warning: No face detected on frame {str(i)}, skipping") 
                cropped_images_list.append(np.zeros((256, 256, 3), dtype=np.uint8))
                crop_info_list.append(None)
                # f_s_list.append(None)
                # x_s_list.append(None)
                # source_info.append(None)
                # source_rot_list.append(None)
        
            # Update progress bar
            pbar.update(1)
        ####  "crop_info" contains the information of the crop
        ####     ['M_o2c'] Matrix of the transformation from source image to crop
        ####     ['M_c2o'] Matrix of the transformation from crop to source image
        ####     ['pt_crop'] keypoints of the cropped image. 
        # If using insightFace, there are 103 keypoints. If using faceAlignment, there are 68. 
        # If using MediaPipe, there are 478. Yes, that's correct, it's really 478.
        # 如果使用insightFace，关键点有103个。如果使用faceAlignment,则有68个。如果使用MediaPipe，则有478个。是的，就是这么多，真是478个。
        ####     ['lmk_source'] 标准化的203个关键点，在sourceImage上的位置
        ####     ['lmk_crop'] 标准化的203个关键点，在cropped Image上的位置
        ####     ['input_image_size'] source image size
        ####     ['dsize'] cropped image size--
        cropped_tensors_out = (
            torch.stack([torch.from_numpy(np_array) for np_array in cropped_images_list])
            / 255
        )
        
        crop_info_dict = {
            'crop_info_list': crop_info_list,
            # 'source_rot_list': source_rot_list,
            # 'f_s_list': f_s_list,
            # 'x_s_list': x_s_list,
            # 'source_info': source_info
        }
        #print(crop_info_dict)
        return (cropped_tensors_out, crop_info_dict)
    
class FaceShaperComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "source_image": ("IMAGE",),
            "cropped_image": ("IMAGE",),
            "crop_info": ("CROPINFO", ),
            },
            # "mismatch_method": (
            #         [
            #             "constant",
            #             "cycle",
            #             "mirror",
            #             "cut"
            #         ],
            #         {"default": "constant"},
            #     ),
            "optional": {
                "mask": ("MASK", {"default": None}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "full_images",
        "mask",
    )
    FUNCTION = "process"
    CATEGORY = "FaceShaper"

    def process(self, source_image, cropped_image, crop_info, mask=None):
        mm.soft_empty_cache()
        gc.collect()
        device = mm.get_torch_device()
        if mm.is_device_mps(device): 
            device = torch.device('cpu') #this function returns NaNs on MPS, defaulting to CPU

        B, H, W, C = source_image.shape
        source_image = source_image.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W
        #cropped_image = cropped_image.permute(0, 3, 1, 2)

        if mask is not None:
            if len(mask.size())==2:
                crop_mask = mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
            else:    
                crop_mask = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        else:
            log.info("Using default mask template")
            crop_mask = cv2.imread(os.path.join(script_directory, "liveportrait", "utils", "resources", "mask_template.png"), cv2.IMREAD_COLOR)
            crop_mask = torch.from_numpy(crop_mask)
            crop_mask = crop_mask.unsqueeze(0).float() / 255.0

        #crop_info = liveportrait_out["crop_info"]
        composited_image_list = []
        out_mask_list = []

        total_frames = len(crop_info["crop_info_list"])
        log.info(f"Total frames: {total_frames}")

        pbar = comfy.utils.ProgressBar(total_frames)
        for i in tqdm(range(total_frames), desc='Compositing..', total=total_frames):
            safe_index = min(i, len(crop_info["crop_info_list"]) - 1)

            #if mismatch_method == "cut":
            source_frame = source_image[safe_index].unsqueeze(0).to(device)
            #print("source_frame.shape", source_frame.shape)
            #else:
            #    source_frame = _get_source_frame(source_image, i, mismatch_method).unsqueeze(0).to(device)

            croppedImage = cropped_image[safe_index].unsqueeze(0).to(device)
            #print("cropped_image.shape", cropped_image.shape)
            #if not :
            #    composited_image_list.append(source_frame.cpu())
            #    out_mask_list.append(torch.zeros((1, 3, H, W), device="cpu"))
            #else:
            #cropped_image = torch.clamp(liveportrait_out["out_list"][i]["out"], 0, 1).permute(0, 2, 3, 1)

            # Transform and blend             
            cropped_image_to_original = _transform_img_kornia(
                croppedImage,
                crop_info["crop_info_list"][safe_index]["M_c2o"],
                dsize=(W, H),
                device=device
                )
            #print("cropped_image_to_original.shape", cropped_image_to_original.shape)
            mask_ori = _transform_img_kornia(
                crop_mask[0].unsqueeze(0),
                crop_info["crop_info_list"][safe_index]["M_c2o"],
                dsize=(W, H),
                device=device
                )
            #print("mask_ori.shape", mask_ori.shape)
            cropped_image_to_original_blend = torch.clip(
                    mask_ori * cropped_image_to_original + (1 - mask_ori) * source_frame, 0, 1
                    )

            composited_image_list.append(cropped_image_to_original_blend.cpu())
            out_mask_list.append(mask_ori.cpu())

            pbar.update(1)

        full_tensors_out = torch.cat(composited_image_list, dim=0)
        full_tensors_out = full_tensors_out.permute(0, 2, 3, 1)

        mask_tensors_out = torch.cat(out_mask_list, dim=0)
        mask_tensors_out = mask_tensors_out[:, 0, :, :]
        
        return (
            full_tensors_out.float(), 
            mask_tensors_out.float()
            )
    
class FaceShaperMatchV2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "source_image": ("IMAGE",),
            "source_crop_info": ("CROPINFO", ),    
            "target_crop_info": ("CROPINFO", ),            
            "landmarkType": (["ALL","OUTLINE"], ),
            "AlignType":(["Width","Height","Landmarks"], ),
            },  
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("Image1","LandmarkImg")
    FUNCTION = "run"

    CATEGORY = "FaceShaper"

    def run(self,source_image,source_crop_info, target_crop_info,landmarkType,AlignType):

    
            tensor1 = source_image*255
            tensor1 = np.array(tensor1, dtype=np.uint8)
            output=[]
            image1 = tensor1[0]
            #image2 = tensor2[0]

            height,width = image1.shape[:2]
            w=width
            h=height
            landmarks1 = source_crop_info["crop_info_list"][0]['lmk_crop']
            landmarks2 = target_crop_info["crop_info_list"][0]['lmk_crop']
            #print(len(landmarks1))
            #len(lendmarks1) will always be 203
            #V2版本改了检测工具，因此只用203个点
            if landmarkType == "ALL" or AlignType == "Landmarks":
                leftEye1=np.mean( landmarks1[0:23],axis=0)
                rightEye1=np.mean( landmarks1[24:47],axis=0)
                leftEye2=np.mean( landmarks2[0:23],axis=0)
                rightEye2=np.mean( landmarks2[24:47],axis=0)
                jaw1=landmarks1[108:144]
                jaw2=landmarks2[108:144]
                centerOfJaw1=np.mean( jaw1,axis=0)
                centerOfJaw2=np.mean( jaw2,axis=0)
            else:
                landmarks1 = landmarks1[108:144]
                landmarks2 = landmarks2[108:144]

            #画面划分成16*16个区域，然后去掉边界框以外的区域。
            src_points = np.array([
                [x, y]
                for x in np.linspace(0, w, 16)
                for y in np.linspace(0, h, 16)
            ])
            
            #上面这些区域同时被加入src和dst，使这些区域不被拉伸（效果是图片边缘不被拉伸）
            src_points = src_points[(src_points[:, 0] <= w/8) | (src_points[:, 0] >= 7*w/8) |  (src_points[:, 1] >= 7*h/8)| (src_points[:, 1] <= h/8)]
            #mark_img = self.draw_landmarks(mark_img, src_points, color=(255, 0, 255))
            dst_points = src_points.copy()

            #不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标
            dst_points = np.append(dst_points,landmarks1,axis=0)

            #变形目标人物的landmarks，先计算边界框
            landmarks2=np.array(landmarks2)
            min_x = np.min(landmarks2[:, 0])
            max_x = np.max(landmarks2[:, 0])
            min_y = np.min(landmarks2[:, 1])
            max_y = np.max(landmarks2[:, 1])
            #得到目标人物的边界框的长宽比
            ratio2 = (max_x - min_x) / (max_y - min_y)

            #变形原始人物的landmarks，边界框
            landmarks1=np.array(landmarks1)
            min_x = np.min(landmarks1[:, 0])
            max_x = np.max(landmarks1[:, 0])
            min_y = np.min(landmarks1[:, 1])
            max_y = np.max(landmarks1[:, 1])
            #得到原始人物的边界框的长宽比以及中心点
            ratio1 = (max_x - min_x) / (max_y - min_y)
            middlePoint = [ (max_x + min_x) / 2, (max_y + min_y) / 2]

            landmarks1_cpy = landmarks1.copy()

            if AlignType=="Width":
            #保持人物脸部边界框中心点不变，垂直方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致
                landmarks1_cpy[:, 1] = (landmarks1_cpy[:, 1] - middlePoint[1]) * ratio1 / ratio2 + middlePoint[1]
            elif AlignType=="Height":
            #保持人物脸部边界框中心点不变，水平方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致
                landmarks1_cpy[:, 0] = (landmarks1_cpy[:, 0] - middlePoint[0]) * ratio2 / ratio1 + middlePoint[0]
            elif AlignType=="Landmarks":
                MiddleOfEyes1 = (leftEye1+rightEye1)/2
                MiddleOfEyes2 = (leftEye2+rightEye2)/2
                distance1 =  ((leftEye1[0] - rightEye1[0]) ** 2 + (leftEye1[1] - rightEye1[1]) ** 2) ** 0.5
                distance2 =  ((leftEye2[0] - rightEye2[0]) ** 2 + (leftEye2[1] - rightEye2[1]) ** 2) ** 0.5
                factor = distance1 / distance2
                MiddleOfEyes2 = np.array(MiddleOfEyes2)
                landmarks1_cpy = (landmarks2 - MiddleOfEyes2) * factor + MiddleOfEyes1

                centerOfJaw2 = np.array(centerOfJaw2)
                jawLineTarget = (landmarks2[108:144] - centerOfJaw2) * factor + centerOfJaw1
                landmarks1_cpy[108:144] = jawLineTarget
            #不知道原作者为何把这个数组叫src，其实这是变形后的坐标
            src_points = np.append(src_points,landmarks1_cpy,axis=0)
            #print(landmarks1_cpy)
            
            mark_img = draw_pointsOnImg(image1, dst_points, color=(255, 255, 0),radius=4)
            mark_img = draw_pointsOnImg(mark_img, src_points, color=(255, 0, 0),radius=3)
            mark_img = drawLineBetweenPoints(mark_img, dst_points,src_points)
            #mark_img = draw_landmarks(image1, landmarks1, color=(255, 0, 0),radius=3)
            #mark_img2 = draw_landmarks(image1, source_crop_info["crop_info_list"][0]['lmk_crop'], color=(255, 0, 0),radius=3)
            # Create the RBF interpolator instance            
            #Tried many times, finally find out these array should be exchange w,h before go into RBFInterpolator            
            src_points[:, [0, 1]] = src_points[:, [1, 0]]
            dst_points[:, [0, 1]] = dst_points[:, [1, 0]]


            rbfy = RBFInterpolator(src_points,dst_points[:,1],kernel="thin_plate_spline")
            rbfx = RBFInterpolator(src_points,dst_points[:,0],kernel="thin_plate_spline")

            # Create a meshgrid to interpolate over the entire image
            img_grid = np.mgrid[0:height, 0:width]

            # flatten grid so it could be feed into interpolation
            flatten=img_grid.reshape(2, -1).T

            # Interpolate the displacement using the RBF interpolators
            map_y = rbfy(flatten).reshape(height,width).astype(np.float32)
            map_x = rbfx(flatten).reshape(height,width).astype(np.float32)
            # Apply the remapping to the image using OpenCV
            warped_image = cv2.remap(image1, map_y, map_x, cv2.INTER_LINEAR)


            warped_image = torch.from_numpy(warped_image.astype(np.float32) / 255.0).unsqueeze(0)               
            mark_img = torch.from_numpy(mark_img.astype(np.float32) / 255.0).unsqueeze(0)  
            #mark_img2= torch.from_numpy(mark_img2.astype(np.float32) / 255.0).unsqueeze(0)  
            output.append(warped_image)
            output.append(mark_img)
            #output.append(mark_img2)
            #print("warped_image",warped_image.shape)
            #print("mark_img",mark_img.shape)
    
            return (output)
    
class FaceShaperFaceMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_info": ("CROPINFO", {"default": []}),
                #"OnSource": ("BOOLEAN", {"default": False}),
                "MaskSize": (["Source","Crop"], ),
                #"image": ("IMAGE", ),
                #"area": (["face", "main_features", "eyes", "left_eye", "right_eye", "nose", "mouth", "face+forehead (if available)"], ),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "grow_tapered": ("BOOLEAN", { "default": False }),
                "blur": ("INT", { "default": 13, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "FaceShaper"

    def segment(self, crop_info, MaskSize,grow, grow_tapered, blur):

#        if (OnSource):
        if MaskSize=="Source":
            lmk = crop_info["crop_info_list"][0]['lmk_source']
            height, width = crop_info["crop_info_list"][0]['input_image_size']
        else:
            lmk = crop_info["crop_info_list"][0]['lmk_crop']
            height = width = crop_info["crop_info_list"][0]['dsize']
        
        ####     ['lmk_source'] 标准化的203个关键点，在sourceImage上的位置
        ####     ['lmk_crop'] 标准化的203个关键点，在cropped Image上的位置
        ####     ['input_image_size'] source image size（二维）
        ####     ['dsize'] cropped image size (就一个数，crop后的图是正方形的)

        #out_mask = []

        mask = mask_from_landmarks(height,width, lmk)
        mask = image_to_tensor(mask).unsqueeze(0).squeeze(-1).clamp(0, 1)
        _, y, x = torch.where(mask)
        x1, x2 = x.min().item(), x.max().item()
        y1, y2 = y.min().item(), y.max().item()
        smooth = int(min(max((x2 - x1), (y2 - y1)) * 0.2, 99))
        if smooth > 1:
            if smooth % 2 == 0:
                smooth+= 1
            mask = T.functional.gaussian_blur(mask.bool().unsqueeze(1), smooth).squeeze(1).float()
        
        if grow != 0:
            mask = expand_mask(mask, grow, grow_tapered)

        if blur > 1:
            if blur % 2 == 0:
                blur+= 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1).float()

        #print("A -- mask.shape",mask.shape)
        # A -- mask.shape torch.Size([1, 512, 512])

        return (mask,)
    ##下面的留着以后改造成同时处理多个crop_info
        mask = mask.squeeze(0).unsqueeze(-1)        
        #print("B -- mask.shape",mask.shape)
        # B -- mask.shape torch.Size([512, 512, 1])        
        out_mask.append(mask)        
        out_mask = torch.stack(out_mask).squeeze(-1)

        print("out_mask.shape", out_mask.shape)
        #out_mask.shape torch.Size([1, 512, 512])
        print("A -- mask.shape",mask.shape)
        return (out_mask,) 
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FaceShaper": FaceShaper,
    "FaceShaperModels": FaceShaperModels,
    "FaceAlignmentCropper": FaceAlignmentCropper,
    "FaceShaperCropper": FaceShaperCropper,
    "FaceShaperShowLandMarks": FaceShaperShowLandMarks,
    "FaceShaperComposite":FaceShaperComposite,
    "FaceShaperLoadInsightFaceCropper":FaceShaperLoadInsightFaceCropper,
    "FaceShaperLoadMediaPipeCropper":FaceShaperLoadMediaPipeCropper,
    "FaceShaperMatchV2":FaceShaperMatchV2,
    "FaceShaperFaceMask":FaceShaperFaceMask,

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "FaceShaper": "FaceShape Match(legacy)",
     "FaceShaperModels":" faceShaper LoadModel DLib(legacy)",
     "FaceAlignmentCropper": "FaceShaper Load FaceAlignment",
     "FaceShaperCropper": "FaceShaper Cropper",
     "FaceShaperShowLandMarks": "FaceShaper Showlandmarks",
     "FaceShaperComposite": "FaceShaper Composite",
     "FaceShaperLoadInsightFaceCropper": "FaceShaper Load InsightFace",
     "FaceShaperLoadMediaPipeCropper": "FaceShaper Load MediaPipe",
     "FaceShaperV2": "FaceShape Match V2",
     "FaceShaperFaceMask":"FaceShaper Face Mask"
}
