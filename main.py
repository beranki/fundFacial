from website import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)


#from recog_assets import *
"""
def run_model(name, train_data):
    lr = 1e-4
    epochs = 50

    run(train_data, epochs, lr, name, save_model=True)

def load_model(name):
    model = keras.models.load_model("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\models\\face_recog_" + name + ".h5", 
                                    custom_objects = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    return model

def verify(model, name):
    results = []
    input_img = preprocess("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\input_imgs\\" + "\\" + name + "\\" + f"input_img_{name}.jpg")
    for img in os.listdir("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\verif_imgs\\" + name):
        valid_img = preprocess("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\verif_imgs\\" + name + "\\" + img)
        result = model.predict(list(np.expand_dims([valid_img, input_img], axis = 1)))
        results.append(result)

    detection = 0
    for result in results:
        if (result > 0.5): detection+=1
    
    return detection/len(results) > 0.5

from register.register import detect_face

def verify_current_image(model):
    cap = cv2.VideoCapture(0)

    ret, prev = cap.read()
    dnn_model = cv2.dnn.readNetFromCaffe("C:/Users/anjan/Documents/messingaroundDL/cheekypeeky/caffeModel/deploy.prototxt.txt", 
    "C:/Users/anjan/Documents/messingaroundDL/cheekypeeky/caffeModel/res10_300x300_ssd_iter_140000.caffemodel")
    cf = 0

    while True:
        _, frame = cap.read()

        blob1 = cv2.dnn.blobFromImage(cv2.resize(prev, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_model.setInput(blob1)
        detections = dnn_model.forward() 

        sX, sY, eX, eY, frame, cf = detect_face(prev, frame, detections, cf)
        if (sX != -1):
            cf = 0
            croppedimg = frame[sY:eY, sX:eX]
            croppedimg = cv2.resize(croppedimg, (105, 105))
            cv2.imwrite("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\input_imgs\\" + name + "\\input_img_"+ name + ".jpg", croppedimg)    

        cv2.imshow("Frame", frame)

        prev = frame
        frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

    verified = verify(model, name)
    if (verified): print(f"Successfully recognized. Hello {name}!")
    else: print(f"Not recognized. ")
#=================================================================================================

name = input("What is your name? ") #- temporary!! please remove when integrating with flask
if name not in os.listdir("C:\\Users\\anjan\Documents\\messingaroundDL\\cheekypeeky\\recog\\recog_assets\\imgs\\anchor_imgs"):
    print("Not valid user")
    exit()

train_data, test_data = init_datasets(name)
train_data, test_data = train_data.shuffle(len(list(train_data))), test_data.shuffle(len(list(test_data)))

if (os.path.exists("C:\\Users\\anjan\\Documents\\messingaroundDL\\cheekypeeky\\recog\\models\\face_recog_" + name + ".h5") is False): 
    run_model("Bhargav", train_data=train_data)
else:
    model = load_model(name)
    verify_current_image(model)
"""
