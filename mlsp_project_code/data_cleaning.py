import numpy as np
import cv2
import dlib 
import imutils
import threading
import multiprocessing
from joblib import Parallel, delayed
from collections import OrderedDict

# Path to file containing the file list
# for all images
FILE_PATH = "vggface_list.txt"

OUTPUT_PATH = 'unaligned.txt'
# For testing
PATH =  'vggface/n000785/0005_01.jpg'

# Face landmark predictor - place in 
# same folder as the file
PREDICTOR = "predictor.dat"

# To view the sketch
VIEW = 0

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

# Cv2 and dlib rectangle conversions
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

# Return type of predictor - shape to a numpy array
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

# Apply gaussian filter to the detected face
# and smoothen out the edges
def sketch(image):
    img_blur = cv2.GaussianBlur(image, (21,21), 0, 0)
    img_blend = cv2.divide(image, img_blur, scale=256)
    dst = cv2.edgePreservingFilter(img_blend, flags=1, sigma_s=1, sigma_r=0.001)
    dst = cv2.detailEnhance(dst, sigma_s=1, sigma_r=0.001)
    return dst

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)
        return output

    def landmark(self,image,gray,draw,rects):
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
            (x, y, w, h) = rect_to_bb(rect)
            for (x, y) in shape:
                #cv2.circle(white, (x, y), 2, (0, 0, 255), -1)
                for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
                    (j, k) = FACIAL_LANDMARKS_IDXS[name]
                    pts = shape[j:k]
                    for l in range(1, len(pts)):
                        ptA = tuple(pts[l - 1])
                        ptB = tuple(pts[l])    
                        cv2.line(draw, ptA, ptB, (0,0,255), 2)
        return draw

def generate_sketch_util(image):
    
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR)
    rects = detector(gray, 1)
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    faceAligned = None
    # Align the face so that eye landmarks are horizontal
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(x < 0 or y < 0 or w < 0 or h < 0):
            return None,None,None
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256,height = 256)
        faceAligned = fa.align(image, gray, rect) 
    # Return None if cant align image
    if(faceAligned is None):
        return None,None,None

    if(VIEW):
        cv2.imshow("Before Crop", faceAligned)
    
    gray_detec = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
    rects_detec = detector(gray_detec, 1)

    # Alignment gives a bigger box around the face including the 
    # shoulders and hands sometimes, so crop the aligned image to face
    if(rects_detec is not None):
        for rect in rects_detec:
            (x, y, w, h) = rect_to_bb(rect) 
            if(x < 0 or y < 0 or w < 0 or h < 0):
                break
            faceAligned = imutils.resize(faceAligned[y:y + h, x:x + w], width=256,height = 256) 
    if(VIEW):
        cv2.imshow("Out",faceAligned)
    
    rects2 = detector(faceAligned, 1)
    gray_aligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

    base_image = faceAligned
    out2 = faceAligned
    # Draw landmarks
    out1 = fa.landmark(faceAligned,gray_aligned,base_image,rects2)

    white = np.zeros([256,256,3],dtype=np.uint8)
    white.fill(255)
    base_image = white

    out0 = fa.landmark(faceAligned,gray_aligned,base_image,rects2)
    
    sk0 = sketch(out0)
    sk1 = sketch(out1)
    sk2 = sketch(out2)
    
    sk0 = cv2.cvtColor(sk0, cv2.COLOR_BGR2GRAY)
    sk1 = cv2.cvtColor(sk1, cv2.COLOR_BGR2GRAY)
    sk2 = cv2.cvtColor(sk2, cv2.COLOR_BGR2GRAY)

    return sk0,sk1,sk2

def generate_sketch(path):
    image = cv2.imread(path)
    if(image is None):
        print("Incorrect path")
        return None

    sk0,sk1,sk2 = generate_sketch_util(image)
    # if(faceSketch is None):
    #     print("Could not align the image" + path)
    return sk0,sk1,sk2

def check_nonaligned_helper(image_path):
    sk,_,_ = generate_sketch(image_path)
    if(sk is None):
        image_path = image_path.lstrip('vggface/')
        return image_path

def check_nonaligned(file_path,output_path):
    extension = 'vggface/'
    unaligned = []
    num_cores = multiprocessing.cpu_count()
    with open(file_path,encoding = 'utf-8') as f:
        unaligned = Parallel(n_jobs=num_cores)(delayed(check_nonaligned_helper)(extension + image_path.strip('\n')) for image_path in f)
    with open(output_path, 'a') as op:
        for img in unaligned:
            if(img is not None):
                op.write(img+'\n')

# Type = 0 - draw landmarks on a blank white base
# Type = 1 - draw landmarks on the sketched face
# Type = 2 - just sketch, no landmarks

def thread_handler(image_path):
    sk0,sk1,sk2 = generate_sketch(image_path)
    image_path = image_path.rstrip(".jpg")
    if(sk0 is None):
        return
    cv2.imwrite(image_path+"_blank.jpg",sk0)
    cv2.imwrite(image_path+"_landmarks.jpg",sk1)
    cv2.imwrite(image_path+"_sketch.jpg",sk2)

def generate(file_path):
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    extension = 'vggface/'
    processes = []
    with open(file_path,encoding = 'utf-8') as f:
        Parallel(n_jobs=num_cores)(delayed(thread_handler)(extension + image_path.strip('\n')) for image_path in f)

check_nonaligned(FILE_PATH,OUTPUT_PATH)
generate(FILE_PATH)
# x,y,z = generate_sketch(PATH)

# cv2.imshow("Sketch1",x)
# cv2.imshow("Sketch2",y)
# cv2.imshow("Sketch3",z)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
