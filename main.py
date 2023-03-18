import pytesseract, cv2, matplotlib.pyplot as plt

PATH_HAAR = 'haarcascade_russian_plate_number.xml'
PATH_DATA = 'Data'
CONFIG = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIGKLMNOPQRSTUVWXY70123456Z89У'
def open_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

    # plt.axis('off') # убираем оси
    # plt.imshow(img)
    # plt.show()

    return img

def detecting_number(img, haar):
    number_rect = haar.detectMultiScale(img, scaleFactor=1.1, minNeighbors = 5)

    for x, y, w, h in number_rect: # координаты прямоугольника (x, y) (x+w, y+h)
        detect_number = img[y+15: y + h-10, x+15:x+w-15] # немного обрезаем

    return detect_number

def enlarge_img(img, percent):
    # .shape - высота ширина каналы
    width = int(img.shape[1] * percent/100)

    height = int(img.shape[0] * percent/100)

    resize_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) 

    return resize_img

def toGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def main():
    img = open_image(f'{PATH_DATA}/3.jpg')
    haar_carcade = cv2.CascadeClassifier(PATH_HAAR)

    img = enlarge_img(img, 105)
    img = detecting_number(img, haar_carcade) # находим координаты номера

    
    plt.axis('off') # убираем оси
    plt.imshow(img, cmap = 'gray')
    plt.show()
    img =  enlarge_img(img, 150) # изменяем размер
    
    img_gray = toGray(img)
    plt.axis('off') # убираем оси
    plt.imshow(img_gray, cmap = 'gray')
    plt.show()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    number = pytesseract.image_to_string(
        img_gray,
        config = CONFIG
    )

    print(number)
if __name__ == '__main__':
    main()
