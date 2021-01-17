import os
import glob
import cv2


images = "Database/"

im = sorted(glob.glob(images + "*.*"), key=os.path.getmtime)

target = "Input/image.jpg"
inp  = cv2.imread(target)


def find_hist(image):
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	return hist

def find_match(hist1, hist2):
	d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
	return d

ta = find_hist(inp)

for i in im:
  j=i
  image = cv2.imread(i)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
  height, width = image.shape[:2]

  for (x, y, w, h) in faces:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = image[ny:ny+nr, nx:nx+nr]
    #cv2.imwrite("image.jpg",faceimg)
  sample = find_hist(faceimg)
  match_result = find_match(sample, ta)
  print("match_result-----------",match_result)
  if (match_result*100)<=90:
  	print("not match")
  else:
  	print("match")
  	image = cv2.imread(j)
  	for (x, y, w, h) in faces:
  		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  	cv2.imshow('matches_result',image)
  	cv2.waitKey(0)
  	cv2.destroyAllWindows()
