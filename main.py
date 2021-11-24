import cv2
import mediapipe as mp
import time
import math

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('Videos/4.mp4')
marks = [11, 12, 23, 24]  # required landmarks
head = [8,6,5,4,0,1,2,3,7,10,9]
l_hand = [18,20,16,22]
r_hand = [15,21,17,19]
l_knee = [26]
r_knee = [25]
boader=250
pTime = 0



def Center(list_x,list_y):
    minX= min(list_x)
    maxX= max(list_x)
    minY= min(list_y)
    maxY= max(list_y)
    C_X = int((minX + maxX) / 2)
    C_Y = int((minY + maxY) / 2)
    cv2.circle(img, (C_X, C_Y), 5, (255, 0, 0), cv2.FILLED)
    return C_X,C_Y

def outFrame(img,C_X,C_Y,width,height,boader):
    h, w, c = img.shape

    W = int(height/2) + boader
    H = int(width/2) + boader
    if C_X + W >= w:
        C_X = w - W
    elif C_X-W <= 0:
        C_X = W
    if C_Y + H >= h:
        C_Y = h - H
    elif C_Y-H <= 0:
        C_Y = H
    cropped_img = img[(C_Y - H):(C_Y + H), (C_X - W): (C_X + W)]
    #cv2.circle(img, ((C_X - W), (C_Y - H)), 5, (255, 0, 0), cv2.FILLED)
    #cv2.circle(img, ((C_Y + H), (C_X + W)), 5, (255, 0, 0), cv2.FILLED)
    print(h, w,C_X-W, C_X+W, C_Y-H, C_Y+H)
    return cropped_img

def findouter(cap):
    allx = []  # all contour x
    ally = []  # all contour y
    while True:     # find maximum contour
        success, img = cap.read()
        if not success:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        lmX = []
        lmY = []
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                lmX.append(cx)
                lmY.append(cy)
            minX = min(lmX)
            minY = min(lmY)
            maxX = max(lmX)
            maxY = max(lmY)
            allx.append(int(maxX-minX))
            ally.append(int(maxY - minY))
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    return allx, ally

allx,ally = findouter(cap)
maxWidth = max(allx)
maxHeight = max(ally)


#maxHeight = 1484
#maxWidth = 992
print(maxHeight ,maxWidth)
cap = cv2.VideoCapture('Videos/4.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'FMP4')
#out = cv2.VideoWriter('out.mp4', fourcc, 30.0, (maxWidth + 2 * boader, maxHeight + 2 * boader))
frame = 0
test=[]
while True:
    success, img = cap.read()
    if not success:
        break
    #width, height = 1280, 720
    #img= cv2.resize(img,(width , height))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    lmList_x = []
    lmList_y = []
    headX = []
    headY = []
    l_handX = []
    l_handY = []
    r_handX = []
    r_handY = []
    l_kneeX = []
    l_kneeY = []
    r_kneeX = []
    r_kneeY = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id in marks:
                lmList_x.append(cx)
                lmList_y.append(cy)
                #cv2.circle(img, (cx,cy), 5, (255, 0, 0), cv2.FILLED)
            if id in head:
                headX.append(cx)
                headY.append(cy)
            if id in l_hand:
                l_handX.append(cx)
                l_handY.append(cy)
            if id in r_hand:
                r_handX.append(cx)
                r_handY.append(cy)
            if id in l_knee:
                l_kneeX.append(cx)
                l_kneeY.append(cy)
            if id in r_knee:
                r_kneeX.append(cx)
                r_kneeY.append(cy)
        head_val2 = [sum(headX) // len(headX), sum(headY) // len(headY)]
        l_hand_val2 = [sum(l_handX) // len(l_handX), sum(l_handY) // len(l_handY)]
        r_hand_val2 = [sum(r_handX) // len(r_handX), sum(r_handY) // len(r_handY)]
        l_knee_val2 = [sum(l_kneeX) // len(l_kneeX), sum(l_kneeY) // len(l_kneeY)]
        r_knee_val2 = [sum(r_kneeX) // len(r_kneeX), sum(r_kneeY) // len(r_kneeY)]

        frame += 1
        if frame != 1:
            C_X, C_Y = Center(lmList_x, lmList_y)
            #C_X = C_X + (head_val2[0] - head_val1[0]) * math.exp(-abs(head_val2[0] - head_val1[0])/10)
            #+ (r_hand_val2[0] - r_hand_val1[0]) * math.exp(-abs(r_hand_val2[0] - r_hand_val1[0]) / 5)
            #- (l_hand_val2[0] - l_hand_val1[0]) * math.exp(-abs(l_hand_val2[0] - l_hand_val1[0]) / 5)
            #+ (r_knee_val2[0] - r_knee_val1[0]) * math.exp(-abs(r_knee_val2[0] - r_knee_val1[0]) / 3)
            #- (l_knee_val2[0] - l_knee_val1[0]) * math.exp(-abs(l_knee_val2[0] - l_knee_val1[0]) / 3)

            #C_Y = C_Y + (head_val2[1] - head_val1[1]) * math.exp(-abs(head_val2[1] - head_val1[1]) / 10)
            #+ (r_hand_val2[1] - r_hand_val1[1]) * math.exp(-abs(r_hand_val2[1] - r_hand_val1[1]) / 5)
            #- (l_hand_val2[1] - l_hand_val1[1]) * math.exp(-abs(l_hand_val2[1] - l_hand_val1[1]) / 5)
            #+ (r_knee_val2[1] - r_knee_val1[1]) * math.exp(-abs(r_knee_val2[1] - r_knee_val1[1]) / 2)
            #- (l_knee_val2[1] - l_knee_val1[1]) * math.exp(-abs(l_knee_val2[1] - l_knee_val1[1]) / 2)

            C_Y = int(C_Y)
            C_X = int(C_X)
        else:
            C_X, C_Y = Center(lmList_x, lmList_y)
        head_val1 = head_val2
        l_hand_val1 = l_hand_val2
        r_hand_val1 = r_hand_val2
        l_knee_val1 = l_knee_val2
        r_knee_val1 = r_knee_val2

        img = outFrame(img,C_X,C_Y,maxHeight,maxWidth,boader)
        #print(img.shape)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", img)
    #out.write(img)
    cv2.waitKey(0)

cap.release()
#out.release()
cv2.destroyAllWindows()
