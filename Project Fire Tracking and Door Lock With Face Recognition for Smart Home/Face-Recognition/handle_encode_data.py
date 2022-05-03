import face_recognition
import os
import cv2
import random

def updateEncodeList(ROOT='data'):
	encodeList = []
	encodeListName = []
	member_list = list(os.listdir(ROOT))
	member_list.remove('config.yml')
	member_list.remove('encode_list.txt')
	member_list.remove('encode_names.txt')
	for member in member_list:
		imgs = os.listdir(os.path.join(ROOT, member))
		for img in imgs:
			images = face_recognition.load_image_file(os.path.join(ROOT, member, img))
			images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
			encodeImg = face_recognition.face_encodings(images)[0]
			print(img)
			encodeList.append(encodeImg)
			encodeListName.append(member)
	
	temp = list(zip(encodeList, encodeListName))
	random.shuffle(temp)
	encodeList, encodeListName = zip(*temp)
	encodeList, encodeListName = list(encodeList), list(encodeListName)
	print(f'len(encodeList) : {len(encodeList)}')
	print(f'encodeListName : {encodeListName}')
	print("UPDATE ENCODE LIST DONE !!")
	return encodeList, encodeListName

