IS_DLIB_INSTALLED = False
try:
    import dlib
    IS_DLIB_INSTALLED = True
except ImportError:
    pass

import torch
import numpy as np
import comfy.utils

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import folder_paths
import os


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
    CATEGORY = "Pano"

    def load_models(self, DetectType):
        out = {}

        # if library == "insightface":
        #     out = InsightFace(provider)
        # else:
        #     out = DLib()
        out = DLib(DetectType)
        return (out, )
           
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
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FaceShaper": FaceShaper,
    "FaceShaperModels": FaceShaperModels
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "FaceShaper": "Face Shape Match",
     "FaceShaperModels":" faceShaper LoadModel"
}
