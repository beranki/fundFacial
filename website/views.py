from ast import Str
from json import load
import multiprocessing
import cv2, os, random
import numpy as np
import warnings
from flask import (
    Blueprint,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    jsonify,
    Response,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from .models import User, Transaction
from website.models import db
warnings.filterwarnings("ignore")

b_views = Blueprint("views", __name__)
current_frame = None
selected_transaction = None
stop_verif_cam = False

@b_views.route("/")
def home():
    if current_user.is_authenticated:
        if (
            os.path.exists(
                "recog\\models\\"
                + str(current_user.id)
                + "\\face_recog.h5"
            )
            is False
        ):
            if p is None or not p.is_alive():
                return redirect(url_for("views.epoch_loading"))

    return render_template("home.html", user=current_user)


@b_views.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user_email = request.form.get("email")
        user_password = request.form.get("password")

        user = User.query.filter_by(email=user_email).first()
        if user:
            if check_password_hash(user.password, user_password):
                flash("Logged in sucessfully.", category="success")
                login_user(user, remember=True)
                return redirect(url_for("views.user_menu"))
            else:
                flash("Incorrect password", category="error")
        else:
            flash("Email does not exist.", category="error")

    return render_template("login.html", user=current_user)


@b_views.route("/signup", methods=["POST", "GET"])
def sign_up():
    print(db)
    if request.method == "POST":
        user_email = request.form.get("email")
        user_password = request.form.get("password")
        conf_password = request.form.get("password_confirm")
        username = request.form.get("username")

        user = User.query.filter_by(email=user_email).first()
        if user:
            flash("Email already registered.", category="error")
        elif user_password != conf_password:
            flash("Passwords are not the same.", category="error")
        else:  #
            new_user = User(
                email=user_email,
                username=username,
                password=generate_password_hash(user_password, "sha256"),
            )
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)

            paths, status = create_user_paths(current_user.id)
            return redirect(url_for("views.collect_face_data"))

    return render_template("signup.html", user=current_user)


# ==================================================================================================================================
cap = cv2.VideoCapture(0)
caffe_model = cv2.dnn.readNetFromCaffe(
    "caffeModel/deploy.prototxt.txt",
    "caffeModel/res10_300x300_ssd_iter_140000.caffemodel",
)
cf = 0
counter = 0

anchor_pac = "recog/recog_assets/imgs/anchor_imgs/"
positive_pac = "recog/recog_assets/imgs/positive_imgs/"
verif_pac = "recog/verif_imgs/"


def depictFlow(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2 + fy**2)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def detect_face(prev_frame, frame, dets, cfincrement):
    flow_frame = None
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        # print(confidence)
        if confidence > 0.5:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            h, w = flow.shape[:2]
            flow_frame = depictFlow(flow)
            if np.average(flow) > 0.1 and np.average(flow) < 0.6:
                cfincrement += 1
            else:
                cfincrement = 0

            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sX, sY, eX, eY) = box.astype("int")
            cv2.rectangle(frame, (sX, sY), (eX, eY), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "C:" + str(int(confidence * 100)) + "%",
                (w - 110, h - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return [sX, sY, eX, eY, frame, cfincrement]
        else:
            return [-1, -1, -1, -1, frame, cfincrement]


def create_user_paths(name):
    global anchor_pac, positive_pac, verif_pac
    paths = []
    try:
        os.mkdir(anchor_pac + str(name))
        os.mkdir(positive_pac + str(name))
        os.mkdir(verif_pac + str(name))
        paths = [
            anchor_pac + str(name),
            positive_pac + str(name),
            verif_pac + str(name),
        ]
    except FileExistsError:
        print("Users already exist")
        return paths, False

    return paths, True


def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def take_user_images(image_paths):
    global caffe_model, cf, counter, cap, current_frame, toggle
    counter = 0
    [
        counter := counter + sum(len(files) for _, _, files in os.walk(rf"{path}"))
        for path in image_paths
    ]
    print(counter)
    ret, prev = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret or not np.any(frame) or not toggle:
            continue

        blob1 = cv2.dnn.blobFromImage(
            cv2.resize(prev, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        caffe_model.setInput(blob1)
        detections = (
            caffe_model.forward()
        )  # detections should have shape [batch size, width, height, channels]

        sX, sY, eX, eY, frame, cf = detect_face(prev, frame, detections, cf)
        if toggle:
            if sX != -1:
                counter += 1
                cf = 0
                croppedimg = frame[sY:eY, sX:eX]
                croppedimg = cv2.resize(croppedimg, (105, 105))

                if counter > 200:
                    cv2.imwrite(
                        image_paths[2] + "/" + str(random.randint(0, 100000)) + ".jpg",
                        croppedimg,
                    )
                else:
                    cv2.imwrite(
                        image_paths[counter % 2]
                        + "/"
                        + str(random.randint(0, 100000))
                        + ".jpg",
                        croppedimg,
                    )

            # cv2.imshow("Frame", frame)
            current_frame = frame.copy()
            flag1, encoded_img = cv2.imencode(".jpg", frame)
            if not flag1:
                continue
            encoded_bytes_img = encoded_img.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded_bytes_img + b"\r\n"
            )

            prev = frame
            frame = cap.read()

        if (
            cv2.waitKey(1) and counter >= 250
        ):  # aiming to have about 100 images in the anchor and positive folder each, and 50 images in the verif_images
            break

    cap.release()
    cv2.destroyAllWindows()

def run_model(name, train_data):
    lr = 1e-4
    epochs = 50

    run(train_data, epochs, lr, name, save_model=True)

def load_model(name):
    model = tf.keras.models.load_model("recog\\models\\face_recog_" + name + ".h5", 
                                    custom_objects = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    return model

def verify(model, name):
    results = []
    input_img = preprocess("recog\\input_imgs\\" + name + "\\" + f"input_img_{name}.jpg")
    for img in os.listdir("recog\\verif_imgs\\" + name):
        valid_img = preprocess("recog\\verif_imgs\\" + name + "\\" + img)
        result = model.predict(list(np.expand_dims([input_img, valid_img], axis = 1)))
        results.append(result)

    detection = 0
    for result in results:
        print(result)
        if (result > 0.5): detection+=1

    print(detection/len(results))

    return detection/len(results) > 0.5

def verify_current_image(user_id):
    global stop_verif_cam, cap, caffe_model

    ret, prev = cap.read()
    cf = 0

    if (os.path.exists(f"recog\\input_imgs\\{user_id}") is False):
        os.mkdir(f"recog\\input_imgs\\{user_id}")

    while True:
        _, frame = cap.read()

        blob1 = cv2.dnn.blobFromImage(cv2.resize(prev, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        caffe_model.setInput(blob1)
        detections = caffe_model.forward() 

        sX, sY, eX, eY, frame, cf = detect_face(prev, frame, detections, cf)
        if (sX != -1):
            cf = 0
            croppedimg = frame[sY:eY, sX:eX]
            croppedimg = cv2.resize(croppedimg, (105, 105))
            cv2.imwrite("recog\\input_imgs\\" + str(user_id) +"\\input_img_"+ str(user_id) + ".jpg", croppedimg)    

            flag1, encoded_img = cv2.imencode(".jpg", frame)
            if not flag1:
                continue
            encoded_bytes_img = encoded_img.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded_bytes_img + b"\r\n"
            )

        prev = frame
        frame = cap.read()

        if cv2.waitKey(1) and stop_verif_cam is True: 
            break

    cap.release()
    cv2.destroyAllWindows()



import tensorflow as tf
import os
import tarfile
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    Input,
    Flatten,
    Dense,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def extract_lfw(path, file, destination):
    ext_file = tarfile.open(path + file)
    ext_file.extractall(path)
    ext_file.close()
    lfw_path = path + "lfw\\"
    for name in os.listdir(lfw_path):
        for i in os.listdir(lfw_path + name):
            os.replace(lfw_path + name + "\\" + i, destination + i)


def preprocess(img_path):
    b_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(b_img)
    img = tf.image.resize(img, (105, 105))
    img = img / 255.0

    return img


def data_augment(paths, num_total):
    data_gen_args = dict(
        rotation_range=90,
        rescale=1.0 / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
    )

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(data_gen_args)

    for path in paths:
        data_flow = data_gen.flow_from_directory(
            directory=path,
            target_size=(105, 105),
            save_to_dir=path + "\\",
            color_mode="rgb",
            class_mode="categorical",
            shuffle=True,
            save_format="jpg",
            batch_size=1,
        )

        for i in range(num_total - len(os.listdir(path + "\\"))):
            data_flow.next()


"""
Everything implemented in this python file follows the Siamese Neural Networks for One Shot Recognition Paper: 
https:\\\\www.cs.cmu.edu\\~rsalakhu\\papers\\oneshot1.pdf
"""

def init_datasets(name):
    data_augment(
        [
            "recog\\recog_assets\\imgs\\positive_imgs\\"
            + name,
            "recog\\recog_assets\\imgs\\anchor_imgs\\"
            + name,
        ],
        100)  # for data augmentation to make sure there's atleast 100
    paths = [
        "recog\\recog_assets\\imgs\\positive_imgs\\" + name,
        "recog\\recog_assets\\imgs\\anchor_imgs\\" + name,
        "recog\\recog_assets\\imgs\\negative_imgs"
    ]

    anchor = tf.data.Dataset.list_files(paths[0] + "\*.jpg").take(100)
    positive = tf.data.Dataset.list_files(paths[1] + "\*.jpg").take(100)
    negative = tf.data.Dataset.list_files(paths[2] + "\*.jpg").take(100)

    positives = tf.data.Dataset.zip(
        (positive, anchor, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
    )
    negatives = tf.data.Dataset.zip(
        (negative, anchor, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
    )
    dataset = positives.concatenate(negatives)

    def preprocess_triplets(valid_img, inp_img, label):
        return (preprocess(valid_img), preprocess(inp_img), label)

    dir = dataset.as_numpy_iterator()
    dataset = dataset.map(preprocess_triplets)
    dataset.shuffle(len(list(dataset)))
    train_ratio = 0.8

    train_data = dataset.take(round(len(dataset) * train_ratio))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = dataset.skip(round(len(dataset) * train_ratio))
    test_data = test_data.take(round(len(dataset) * (1 - train_ratio)))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data, test_data


class L1Dist(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, inp_embed, val_embed):
        return tf.math.abs(inp_embed - val_embed)


class Embedding(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, inp):
        # Block 1
        conv_1 = Conv2D(64, (10, 10), activation="relu")(inp)
        max_1 = MaxPooling2D(64, (2, 2), padding="same")(conv_1)
        # Block 2
        conv_2 = Conv2D(128, (7, 7), activation="relu")(max_1)
        max_2 = MaxPooling2D(64, (2, 2), padding="same")(conv_2)
        # Block 3
        conv_3 = Conv2D(128, (4, 4), activation="relu")(max_2)
        max_3 = MaxPooling2D(64, (2, 2), padding="same")(conv_3)
        # Embedding Block
        conv_4 = Conv2D(256, (4, 4), activation="relu")(max_3)
        flatten1 = Flatten()(conv_4)
        dense1 = Dense(4096, activation="sigmoid")(flatten1)

        return dense1


class SiameseModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        inp_img = Input(name="input_img", shape=(105, 105, 3))
        valid_img = Input(name="valid_img", shape=(105, 105, 3))

        embedding = Embedding()
        valid_embed = embedding(valid_img)
        inp_embed = embedding(inp_img)

        l1_dist = L1Dist()
        dist = l1_dist(inp_embed, valid_embed)

        dense2 = Dense(1, activation="sigmoid")(dist)

        return Model(inputs=[inp_img, valid_img], outputs=[dense2], name="siam_network")


current_epoch = 0


def run(train_data, epochs, lr, name, save_model=True):
    global current_epoch
    # inp = Input(shape = (105, 105, 3), name = "input img")
    # valid = Input(shape = (105, 105, 3), name = "valid img")
    s_model = SiameseModel()
    siam_model = s_model()

    for epoch in range(epochs):
        current_epoch = epoch
        print(f"EPOCH {epoch+1}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(train_data))

        for step, batch in enumerate(train_data):
            opt = tf.keras.optimizers.Adam(lr)
            bcl = tf.losses.BinaryCrossentropy()
            # one step=============================
            with tf.GradientTape() as tape:
                X = batch[:2]  # features
                y = batch[2]  # labels
                y_pred = siam_model(X, training=True)
                loss = bcl(y, y_pred)

            grad = tape.gradient(loss, s_model.trainable_variables)
            opt.apply_gradients(zip(grad, s_model.trainable_variables))
            progbar.update(step + 1)

    if save_model is True:
        to_save = (
            "recog\\models\\"
            + name
            + "\\face_recog_model.h5"
        )
        s_model.save(to_save)

    return loss

# ===============================================================================
from multiprocessing import Process


@b_views.route("/model_training")
@login_required
def model_training():
    return render_template("model_training.html")


@b_views.route("/video_feed")
def video_feed():
    paths = [
        anchor_pac + str(current_user.id),
        positive_pac + str(current_user.id),
        verif_pac + str(current_user.id),
    ]
    return Response(take_user_images(paths), mimetype="multipart/x-mixed-replace; boundary=frame")

@b_views.route("/video_feed_2")
def video_feed_2():
    return Response(verify_current_image(str(current_user.id)), mimetype="multipart/x-mixed-replace; boundary=frame")

def run_model(name, train_data):
    lr = 1e-4
    epochs = 50

    run(train_data, epochs, lr, name, save_model=True)

def task(user_id):
    train_data, test_data = init_datasets(str(user_id))

    train_data, test_data = train_data.shuffle(len(list(train_data))), test_data.shuffle(len(list(test_data)))
    print("\n\n")
    print(train_data)
    print(test_data)
    print("\n\n")
    run_model(user_id, train_data=train_data)

toggle = True
p = None

@b_views.route("/verify_tasks", methods=["GET", "POST"])
def verify_tasks():
    global toggle, cap, counter, p
    if request.method == "POST":
        if request.form.get("photo") == "Take a Photo":
            toggle = not toggle
            print(toggle)
        if request.form.get("proceed") == "Proceed to User Menu" and counter >= 250:
            if (
                os.path.exists("recog\\models\\"
                    + str(current_user.id)
                    + "\\face_recog.h5") is False):

                return redirect(url_for("views.epoch_loading"))
            # else:
            return redirect(url_for("views.user_menu"))
 
    return redirect(url_for("views.collect_face_data"))

@b_views.route("/requests", methods=["GET", "POST"])
def tasks():
    global toggle, cap, counter, p, stop_verif_cam
    if request.method == "POST":
        if request.form.get("stop") == "Start/Stop Recording":
            toggle = not toggle
            print(toggle)
            return redirect(url_for("views.collect_face_data"))
        if request.form.get("proceed") == "Proceed to User Menu" and counter >= 250:
            if (
                os.path.exists("recog\\models\\"
                    + str(current_user.id)
                    + "\\face_recog.h5") is False):

                return redirect(url_for("views.epoch_loading"))
            # else:
            return redirect(url_for("views.user_menu"))

        if request.form.get("capture") == "Capture Input Image":
            stop_verif_cam = True
            print(stop_verif_cam)
            print(selected_transaction)


            model = load_model(str(current_user.id))
            verified = verify(model, str(current_user.id))
            if (verified): 
                print(f"Transaction successfully verified.")
                db.session.delete(selected_transaction)
                db.session.commit()
            else: 
                print("Not recognized.")

            return redirect(url_for("views.user_menu"))


post_reqs = 0

import time

@b_views.route("/track")
def track(user_id):
    start_time = time.time()
    while (os.path.exists("recog\\models\\" + str(user_id)+ "\\face_recog.h5") is False):
        #print("passing..." + str(time.time() - start_time))
        pass

    return redirect(url_for("views.user_menu"))


@b_views.route("/epoch_loading", methods=["POST", "GET"])
@login_required
def epoch_loading():
    global p, post_reqs, current_epoch
    if (request.method == "GET"):
        if (os.path.exists("recog\\models\\face_recog_" + str(current_user.id) + ".h5")):
            return redirect(url_for("views.user_menu"))
    if (request.method == "POST"):
        print(post_reqs)
        post_reqs += 1
        if (post_reqs == 1):
            print(os.path.exists("recog\\recog_assets\\imgs\\anchor_imgs\\" + str(current_user.id)))
            print(p)
            if p is None:
                paths, status = create_user_paths(current_user.id)
                if (status): 
                    return redirect(url_for("views.collect_face_data"))
                else: 
                    p1 = Process(target=task, args=[current_user.id])
                    p2 = Process(target=track, args=[current_user.id])
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()
            

    return render_template("model_training_wait.html", current_epoch=current_epoch, user=current_user)


@b_views.route("/collect_face_data", methods=["POST", "GET"])
@login_required
def collect_face_data():
    return render_template("collect_face_data.html", user=current_user)

@b_views.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out of account.", category="success")
    return redirect(url_for("views.home"))

from flask import get_flashed_messages

@b_views.route("/user_menu", methods=["POST", "GET"])
@login_required
def user_menu():
    get_flashed_messages()
    global p, selected_transaction
    if request.method == "POST":
        print(p)
        if (os.path.exists(
                f"recog\\models\\face_recog_{current_user.id}.h5") is False):
            if (p is not None):
                if (p.is_alive()):
                    flash("Biometric model still training.", category="error")
                else: 
                    p = None
                    print("reached1")
                    return redirect(url_for("views.epoch_loading"))
            else:
                p = None
                print("reached6")
                return redirect(url_for("views.epoch_loading"))
        else:
            button_pressed = list(request.form)[0]
            transaction_id = int(button_pressed.split("_")[-1])
            selected_transaction = db.session.query(Transaction).filter(
                Transaction.id == transaction_id
            )

            print(button_pressed)
            if button_pressed.__contains__("accept"):
                print("a")
                return redirect(url_for("views.verify_request"))

            if button_pressed.__contains__("deny"):
                print("d")
                return redirect(url_for("views.verify_request"))

    incoming_transactions = list(
        db.session.query(Transaction).filter(Transaction.to_id == current_user.id)
    )
    outgoing_transactions = list(
        db.session.query(Transaction).filter(Transaction.from_id == current_user.id)
    )
    # print(Transaction.query.all())

    incoming_data = []
    for incoming in incoming_transactions:
        from_person = User.query.filter_by(id=incoming.from_id).first()
        incoming_data.append([from_person, incoming])

    outgoing_data = []
    for outgoing in outgoing_transactions:
        to_person = User.query.filter_by(id=outgoing.to_id).first()
        outgoing_data.append([to_person, outgoing])

    #print(incoming_data)
    #print(outgoing_data)

    return render_template(
        "user_menu.html",
        user=current_user,
        incoming_data=incoming_data,
        outgoing_data=outgoing_data,
    )


@b_views.route("/verify_request", methods=["GET", "POST"])
@login_required
def verify_request():
    global selected_transaction
    if (selected_transaction is None):
        flash("Didn't properly select a transaction.", category="error")
        return redirect(url_for("views.user_menu"))

    print(selected_transaction)
    return render_template("verify_request.html", user=current_user)
        


@b_views.route("/transaction", methods=["GET", "POST"])
@login_required
def transaction():
    if request.method == "POST":
        amount = request.form.get("amount")
        other_email = request.form.get("other_user")
        status = request.form.get("status")

        to_user = User.query.filter_by(email=other_email).first()
        if to_user:
            logged_in_user = User.query.get(current_user.id)
            if other_email == current_user.email:
                flash("Cannot request from yourself.", category="error")
            elif (
                status.lower() == "gift"
                and amount != ""
                and current_user.balance < float(amount)
            ):
                flash(
                    "The amount given extends your current balance.", category="error"
                )
            else:
                new_transaction = Transaction(
                    amount=amount,
                    from_id=current_user.id,
                    to_id=to_user.id,
                    status=status,
                )
                db.session.add(new_transaction)
                db.session.commit()
                flash("New transaction made!", category="success")
                return redirect(url_for("views.user_menu"))
        else:
            flash("Requested user doesn't exist.", category="error")

    return render_template("transaction.html", user=current_user)


@b_views.route("/user-info")
@login_required
def user_info():
    return render_template("user_info.html", user=current_user)
