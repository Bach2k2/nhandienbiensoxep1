import cv2 # Thư viên Opencv hỗ trợ xử lý hình ảnh
from pathlib import Path # Path và argparse lần lượt là thuư viện hỗ trợ xử lý đường dẫn và tham số dòng lệnh
import argparse
import numpy as np
import urllib.request
import time # Thư viện hỗ trợ tính thời gian thực thi
import os
import matplotlib.pyplot as plt
url = 'http://192.168.91.35/cam-lo.jpg'
#url = 'http://192.168.1.120/cam-lo.jpg'
from src.lp_recognition import E2E # lớp nhận dạng biển số xe trong file lp.recognition

def get_arguments():
    arg = argparse.ArgumentParser() # Khởi tạo đối tượng Ardument Parser
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg') # các tham số cho đối tượng
    #arg = argparse.ArgumentParser('-i', '--image_path', help='link to image', default='./samples/1.jpg') #thay hàm add_argument bằng cách này sẽ gặp bug. Không trùng tham số với hàm
    return arg.parse_args() # hàm parse_args()



end=""
model = E2E() #
#
# # recognize license plate
# image = model.predict(img)  # check hàm predict
# read image
# img = cv2.imread(str(img_path))  # đọc ảnh từ đường dẫn
count=0
def save_img(filename,img):
    cv2.imwrite(filename,img)
print("hello")
while (1):
    start = time.time()  # lấy thời điểm bắt đầu
    img = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_np, -1)
   #  box = [5, 5, 260, 220]
   # #print(frame)
   #  if box is not None:
   #      (x, y, w, h) = box[0], box[1], box[2], box[3]
   #      img = frame[y:h, x:w]
   #      img1 = cv2.resize(img, (224, 224))
   #      img2 = cv2.resize(img, (256, 256))
   #      img_array = np.expand_dims(img1, axis=0)
   #      #pImg = preprocess_input(img_array)
   #
   #      prediction = model.predict(pImg)
   #      prediction = prediction[0]
   #      predicted_class = np.argmax(prediction, axis=-1)
   #      pro = prediction[predicted_class]
   #      # s=str(predicted_class)+"    xsuat"+ str(pro)
   #
   #      s = "Label: {}".format(str(classes[predicted_class]))
   #      s2 = "Pro: {}".format(str(pro))
   #
   #      print(prediction)
   #      print(f"label {s}")
   #      if (pro > 0.5 and predicted_class != 3) or (0.5 < pro < 0.8 and predicted_class == 3):
   #          # .....
   #          cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
   #          cv2.putText(frame, s, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
   #          cv2.putText(frame, s2, (x + 35, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
   #          print(f"predict class: {predicted_class}")
   #          print(f"img2: {img2}")
    image = model.predict(frame)
    cv2.imshow("img", frame)
    # plt.imshow(frame)
    # plt.show()
    print(count)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        # frame.release()
        cv2.destroyAllWindows()
        end = time.time()  # lấy thời điểm kết thúc
        print('Model process on %.2f s' % (end - start))
        break
    elif chr(cv2.waitKey(0) & 255) == 's':
        end = time.time()
        if not os.path.exists("samples"):
            os.makedirs("samples")
            print("Loi thu muc chua ton tai")
        filename="samples/image_"+str(end)+str(count)+".jpg"
        count+=1
        print("Luu anh: ", filename)
        save_img(filename, frame)
        plt.imshow(frame)
        plt.show()
    elif chr(cv2.waitKey(0) & 255) == 'a':
        args = get_arguments()  # trả về đối tượng ảnh
        img_path = Path(args.image_path)  # lấy đường dẫn ảnh từ tham số args. imagepath
        # read image
        img = cv2.imread(str(img_path))  # đọc ảnh từ đường dẫn
        image = model.predict(img)  # check hàm predict
        cv2.imshow("img", image)



# # start
# start = time.time() # lấy thời điểm bắt đầu
#
# # load model
# model = E2E() #
#
# # recognize license plate
# image = model.predict(img)  # check hàm predict
#
# # end
# end = time.time()# lấy thời điểm kết thúc
#

#
# # show image
# cv2.imshow('License Plate', image) # Bao gồm tên cửa sổ và ảnh
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     exit(0)
#
#
# cv2.destroyAllWindows()
