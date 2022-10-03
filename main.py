import math
import dlib
import os
import cv2

def cross_point(x1,y1,x2,y2,x3,y3,x4,y4):
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

faceDetector = dlib.get_frontal_face_detector()
landMarkPred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
pos = []  # 记录人脸的68个地标
box_wide,box_hight = 0,0
#path = "15_0508funnydunkey"
path = "22_0508funnydunkey"
files = os.listdir(path)
for i in range(len(files)):
    #以第一帧来确定框的大小，左眼+眉毛区域以(17的x，19的y)作为左上角，(21的x，40的y+10)为右下角来得到框的大小
    #右眼+眉毛同理，(22的x，24的y)作为左上角，根据框的大小来确定右下角
    #脸颊区域，32  34   嘴巴用 上51 下57 左48 右54 先计算嘴巴那块的中心点，把中心点作为框的中心点，然后确定框的左上角和右下角
    #额头区域，先确定额头区域下边框的中间点  根据23 20俩点来确定
    if i==0:
        print(0)
        img = cv2.imread(path+"/"+files[i])
        img_origin = cv2.imread(path+"/"+files[i])
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImg, 1)
        for face in faces:
            landMarks = landMarkPred(img, face)
            index = 0
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos.append(pt_pos)
                cv2.circle(img, pt_pos, 2, (0, 255), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                index += 1

        # 先裁剪眼睛+眉毛这块区域，确定切割框的大小
        # 左眼+眉毛
        left_x = pos[17][0]
        left_y = pos[19][1]
        right_x = pos[21][0]
        right_y = pos[40][1] + 10
        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (135, 22, 222), 1)
        cv2.imshow('img1', img)
        cv2.waitKey(0)
        leftEye = img_origin[left_y:right_y, left_x:right_x]
        name = files[i].split(".")[0]
        print(name)
        if os.path.exists(path+"_data/"+name)==False:
            os.makedirs(path+"_data/"+name)
        cv2.imwrite(path+"_data/"+name+"/eyeLeft.jpg" , leftEye)

        # 确定盒子大小
        box_wide = right_x - left_x
        box_hight = right_y - left_y
        #print("box wide and hight:", box_wide, box_hight)

        # 右眼+眉毛
        rightEye_upLeftCoor = (pos[22][0], pos[24][1])
        rightEye_right_x = rightEye_upLeftCoor[0] + box_wide
        rightEye_right_y = rightEye_upLeftCoor[1] + box_hight
        cv2.rectangle(img, rightEye_upLeftCoor, (rightEye_right_x, rightEye_right_y), (135, 22, 222), 1)
        cv2.imshow('img2', img)
        cv2.waitKey(0)
        rightEye = img_origin[rightEye_upLeftCoor[1]:rightEye_right_y, rightEye_upLeftCoor[0]:rightEye_right_x]
        cv2.imwrite(path+"_data/"+name+"/eyeRight.jpg" , rightEye)



        # 脸颊两边的区域
        # 脸颊左边
        cheek_left_rightDown = (pos[32][0], pos[32][1])
        cheek_left_leftUp = (cheek_left_rightDown[0] - box_wide, cheek_left_rightDown[1] - box_hight)
        cv2.rectangle(img, cheek_left_leftUp, cheek_left_rightDown, (135, 22, 222), 1)
        cv2.imshow('img3', img)
        cv2.waitKey(0)
        cheek_left = img_origin[cheek_left_leftUp[1]:cheek_left_rightDown[1],
                     cheek_left_leftUp[0]:cheek_left_rightDown[0]]
        cv2.imwrite(path+"_data/"+name+"/cheekLeft.jpg", cheek_left)
        # 脸颊右边
        cheek_right_leftUp = (pos[34][0], pos[34][1] - box_hight)
        cheek_right_rightDown = (pos[34][0] + box_wide, pos[34][1])
        cv2.rectangle(img, cheek_right_leftUp, cheek_right_rightDown, (135, 22, 222), 1)
        cv2.imshow('img4', img)
        cv2.waitKey(0)
        cheek_right = img_origin[cheek_right_leftUp[1]:cheek_right_rightDown[1],
                      cheek_right_leftUp[0]:cheek_right_rightDown[0]]
        cv2.imwrite(path+"_data/"+name+"/cheekRight.jpg", cheek_right)


        # 嘴巴区域  上51 下57 左48 右54 先计算嘴巴那块的中心点，把中心点作为框的中心点，然后确定框的左上角和右下角
        # mouth_center = (pos[48][0]+pos[54][0])/2
        mouth_center = (cross_point(pos[48][0], pos[48][1], pos[54][0], pos[54][1],
                                    pos[51][0], pos[51][1], pos[57][0], pos[57][1]))
        mouth_center[0] = int(mouth_center[0])
        mouth_center[1] = int(mouth_center[1])
        print("mouth_center:", mouth_center)
        mouth_leftUp = (int((mouth_center[0]) - (box_wide / 2)), int(mouth_center[1] - (box_hight / 2)))
        mouth_rightDown = (int((mouth_center[0]) + (box_wide / 2)), int(mouth_center[1] + (box_hight / 2)))
        cv2.rectangle(img, mouth_leftUp, mouth_rightDown, (135, 22, 222), 1)  # 注意rectangle 参数不能是浮点数需要转换为int型
        cv2.imshow('img5', img)
        cv2.waitKey(0)

        mouth = img_origin[mouth_leftUp[1]:mouth_rightDown[1], mouth_leftUp[0]:mouth_rightDown[0]]
        cv2.imshow('mouth', mouth)
        cv2.waitKey(0)
        cv2.imwrite(path+"_data/"+name+"/mouth.jpg", mouth)


        # 额头区域  先确定额头区域下边框的中间点  根据23 20俩点来确定
        print("forehead_down_center:", int((pos[23][0] + pos[20][0]) / 2))
        forehead_down_center = int((pos[23][0] + pos[20][0]) / 2)
        forehead_leftUp = (forehead_down_center - int(box_wide / 2), pos[20][1] - box_hight)
        forehead_rightDown = (forehead_down_center + int(box_wide / 2), pos[20][1])
        cv2.rectangle(img, forehead_leftUp, forehead_rightDown, (135, 22, 222), 1)  # 注意rectangle 参数不能是浮点数需要转换为int型
        cv2.imshow('img6', img)
        cv2.waitKey(0)

        forehead = img_origin[forehead_leftUp[1]:forehead_rightDown[1], forehead_leftUp[0] - 1:forehead_rightDown[0]]
        cv2.imshow('forehead', forehead)
        cv2.waitKey(0)
        cv2.imwrite(path+"_data/"+name+"/forehead.jpg", forehead)
    else:
        pos = []#清空坐标
        img = cv2.imread(path+"/" + files[i])
        img_origin = cv2.imread(path+"/" + files[i])
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImg, 1)
        for face in faces:
            landMarks = landMarkPred(img, face)
            index = 0
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos.append(pt_pos)
                cv2.circle(img, pt_pos, 2, (0, 255), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                index += 1

        #左眼+眉毛
        left_x = pos[17][0]
        left_y = pos[19][1]
        eye_left_rightDown = (left_x+box_wide,left_y+box_hight)
        leftEye = img_origin[left_y:eye_left_rightDown[1], left_x:eye_left_rightDown[0]]
        name = files[i].split(".")[0]
        print(name)
        if os.path.exists(path+"_data/"+ name) == False:
            os.makedirs(path+"_data/" + name)
        cv2.imwrite(path+"_data/" + name + "/eyeLeft.jpg", leftEye)

        #右眼+眉毛
        eye_right_leftUp = (pos[22][0], pos[24][1])
        eye_right_rightDown = (pos[22][0]+box_wide,pos[24][1]+box_hight)
        rightEye = img_origin[eye_right_leftUp[1]:eye_right_rightDown[1], eye_right_leftUp[0]:eye_right_rightDown[0]]
        cv2.imwrite(path+"_data/" + name + "/eyeRight.jpg", rightEye)


        #脸颊区域
        cheek_left_rightDown = (pos[32][0], pos[32][1])
        cheek_left_leftUp = (cheek_left_rightDown[0] - box_wide, cheek_left_rightDown[1] - box_hight)
        cheek_left = img_origin[cheek_left_leftUp[1]:cheek_left_rightDown[1],
                     cheek_left_leftUp[0]:cheek_left_rightDown[0]]
        cv2.imwrite(path+"_data/" + name + "/cheekLeft.jpg", cheek_left)
        # 脸颊右边
        cheek_right_leftUp = (pos[34][0], pos[34][1] - box_hight)
        cheek_right_rightDown = (pos[34][0] + box_wide, pos[34][1])
        cheek_right = img_origin[cheek_right_leftUp[1]:cheek_right_rightDown[1],
                      cheek_right_leftUp[0]:cheek_right_rightDown[0]]
        cv2.imwrite(path+"_data/" + name + "/cheekRight.jpg", cheek_right)



        # 嘴巴区域  上51 下57 左48 右54 先计算嘴巴那块的中心点，把中心点作为框的中心点，然后确定框的左上角和右下角
        # mouth_center = (pos[48][0]+pos[54][0])/2
        mouth_center = (cross_point(pos[48][0], pos[48][1], pos[54][0], pos[54][1],
                                    pos[51][0], pos[51][1], pos[57][0], pos[57][1]))
        mouth_center[0] = int(mouth_center[0])
        mouth_center[1] = int(mouth_center[1])
        print("mouth_center:", mouth_center)
        mouth_leftUp = (int((mouth_center[0]) - (box_wide / 2)), int(mouth_center[1] - (box_hight / 2)))
        mouth_rightDown = (int((mouth_center[0]) + (box_wide / 2)), int(mouth_center[1] + (box_hight / 2)))

        mouth = img_origin[mouth_leftUp[1]:mouth_rightDown[1], mouth_leftUp[0]:mouth_rightDown[0]]
        cv2.imwrite(path+"_data/" + name + "/mouth.jpg", mouth)



        # 额头区域  先确定额头区域下边框的中间点  根据23 20俩点来确定
        print("forehead_down_center:", int((pos[23][0] + pos[20][0]) / 2))
        forehead_down_center = int((pos[23][0] + pos[20][0]) / 2)
        forehead_leftUp = (forehead_down_center - int(box_wide / 2), pos[20][1] - box_hight)
        forehead_rightDown = (forehead_down_center + int(box_wide / 2), pos[20][1])

        forehead = img_origin[forehead_leftUp[1]:forehead_rightDown[1], forehead_leftUp[0] - 1:forehead_rightDown[0]]
        cv2.imwrite(path+"_data/" + name + "/forehead.jpg", forehead)

"""
img = cv2.imread(path+files[i])
img_origin = cv2.imread(path+files[i])
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceDetector(grayImg, 1)
for face in faces:
    landMarks = landMarkPred(img, face)
    index = 0
    for point in landMarks.parts():
        pt_pos = (point.x, point.y)
        pos.append(pt_pos)
        cv2.circle(img, pt_pos, 2, (0, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        index += 1

#cv2.imshow('img', img)
#cv2.waitKey(0)

# 先裁剪眼睛+眉毛这块区域，确定切割框的大小
# 左眼+眉毛
left_x = pos[17][0]
left_y = pos[19][1]
right_x = pos[21][0]
right_y = pos[40][1] + 10
cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (135, 22, 222), 1)
cv2.imshow('img1', img)
cv2.waitKey(0)
leftEye = img_origin[left_y:right_y, left_x:right_x]
name = files[i].split(".")[0]
print(name)
if os.path.exists("data/"+name)==False:
    os.mkdir("data/"+name)
cv2.imwrite("data/"+name+"/eyeLeft.jpg" , leftEye)

# 确定盒子大小
box_wide = right_x - left_x
box_hight = right_y - left_y
#print("box wide and hight:", box_wide, box_hight)


#右眼+眉毛
rightEye_upLeftCoor = (pos[22][0],pos[24][1])
rightEye_right_x = rightEye_upLeftCoor[0]+box_wide
rightEye_right_y = rightEye_upLeftCoor[1]+box_hight
cv2.rectangle(img,rightEye_upLeftCoor,(rightEye_right_x,rightEye_right_y),(135,22,222),1)
cv2.imshow('img2', img)
cv2.waitKey(0)
rightEye = img_origin[rightEye_upLeftCoor[1]:rightEye_right_y,rightEye_upLeftCoor[0]:rightEye_right_x]
cv2.imwrite("data/forehead/4.jpg",rightEye )

#脸颊两边的区域
#脸颊左边
cheek_left_rightDown = (pos[32][0],pos[32][1])
cheek_left_leftUp = (cheek_left_rightDown[0]-box_wide,cheek_left_rightDown[1]-box_hight)
cv2.rectangle(img,cheek_left_leftUp,cheek_left_rightDown,(135,22,222),1)
cv2.imshow('img3', img)
cv2.waitKey(0)
cheek_left = img_origin[cheek_left_leftUp[1]:cheek_left_rightDown[1],cheek_left_leftUp[0]:cheek_left_rightDown[0]]
cv2.imwrite("data/forehead/3.jpg",cheek_left)
#脸颊右边
cheek_right_leftUp = (pos[34][0],pos[34][1]-box_hight)
cheek_right_rightDown = (pos[34][0]+box_wide,pos[34][1])
cv2.rectangle(img,cheek_right_leftUp,cheek_right_rightDown,(135,22,222),1)
cv2.imshow('img4', img)
cv2.waitKey(0)
cheek_right = img_origin[cheek_right_leftUp[1]:cheek_right_rightDown[1],cheek_right_leftUp[0]:cheek_right_rightDown[0]]
cv2.imwrite("data/forehead/2.jpg",cheek_right)

#嘴巴区域  上51 下57 左48 右54 先计算嘴巴那块的中心点，把中心点作为框的中心点，然后确定框的左上角和右下角
#mouth_center = (pos[48][0]+pos[54][0])/2
mouth_center = (cross_point(pos[48][0],pos[48][1],pos[54][0],pos[54][1],
                  pos[51][0],pos[51][1],pos[57][0],pos[57][1]))
mouth_center[0] = int(mouth_center[0])
mouth_center[1] = int(mouth_center[1])
print("mouth_center:",mouth_center)
mouth_leftUp = (int((mouth_center[0])-(box_wide/2)),int(mouth_center[1]-(box_hight/2)))
mouth_rightDown = (int((mouth_center[0])+(box_wide/2)),int(mouth_center[1]+(box_hight/2)))
cv2.rectangle(img,mouth_leftUp,mouth_rightDown,(135,22,222),1)#注意rectangle 参数不能是浮点数需要转换为int型
cv2.imshow('img5', img)
cv2.waitKey(0)

mouth = img_origin[mouth_leftUp[1]:mouth_rightDown[1],mouth_leftUp[0]:mouth_rightDown[0]]
cv2.imshow('mouth', mouth)
cv2.waitKey(0)
cv2.imwrite("data/forehead/1.jpg",mouth)

#额头区域  先确定额头区域下边框的中间点  根据23 20俩点来确定
print("forehead_down_center:",int((pos[23][0]+pos[20][0])/2))
forehead_down_center = int((pos[23][0]+pos[20][0])/2)
forehead_leftUp = (forehead_down_center-int(box_wide/2),pos[20][1]-box_hight)
forehead_rightDown = (forehead_down_center+int(box_wide/2),pos[20][1])
cv2.rectangle(img,forehead_leftUp,forehead_rightDown,(135,22,222),1)#注意rectangle 参数不能是浮点数需要转换为int型
cv2.imshow('img6', img)
cv2.waitKey(0)

forehead = img_origin[forehead_leftUp[1]:forehead_rightDown[1],forehead_leftUp[0]-1:forehead_rightDown[0]]
cv2.imshow('forehead', forehead)
cv2.waitKey(0)
cv2.imwrite("data/forehead/0.jpg",forehead)
"""