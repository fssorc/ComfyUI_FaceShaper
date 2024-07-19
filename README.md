# ComfyUI_FaceShaper
Match two faces' shape before using other face swap nodes

图片换脸的工具在换脸的时候一般都只把五官特征换上，脸型则没有换。当目标人物和原照片里的人物脸型相差很多的时候，换脸结果就不太理想。
本项目是一个小脚本，能按照目标人物的脸部轮廓纵横比例先把原始照片中的人脸进行液化拉伸。得到的结果可以作为其他换脸节点的输入。

Face-swapping tools typically only replace facial features during the swap, without altering the facial shape. When there is a significant difference in facial shape between the target person and the person in the original photo, the result of the face swap is less satisfactory.

This project is a small script that can first liquefy and stretch the face in the original photo according to the horizontal and vertical proportions of the target person's facial contour. The resulting image can be used as input for other face-swapping nodes.

#Install
run in ComfyUI/custom_nodes:
git clone https://github.com/fssorc/ComfyUI_FaceShaper

need dlib and opencv-python
put dlib model files in ComfyUI/models/dlib/
shape_predictor_68_face_landmarks.dat
shape_predictor_81_face_landmarks.dat
shape_predictor_5_face_landmarks.dat

Test Results:
the faceSwap tool I am using is instantid and DZ_FaceDetailer


While writing the code, I was inspired by ComfyUI_FaceAnalysis, and I would like to express my gratitude here.