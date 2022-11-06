# fundFacial
![image](https://user-images.githubusercontent.com/77950550/197428516-190b99f0-c579-444e-b25d-dace67f82509.png)

This project is filled with bugs and inefficient code, but should be used as a marker for my journey into web development. 

Major Problems:
- The code is repetitive in nature and often is slower than needs be, especially in the case of the custom trained siamese model. Instead of referring to a transfer model, for the purpose of understand better how the Siamese Model worked, I custom trained it for each user's data. This is a big mistake and something I realized would be inconvenient early on.
- Second, the data is not secure by any means. It's easily compromised and for an application centered around security, it fails to achieve that in certain aspects.

Cool Things:
- I felt I made an easy to use and innovative interface to handle the transaction requests themselves, and it was definitely a leap out of my comfort zone as I utilized schemas and other essentials to UX dev which I had not prior.

Things to Improve:
- Definitely switching to a transfer model for face recognition...big big mistake to custom train it per user.
- Improve security and continue working towards making it a more user friendly app.
- Automate a lot of the processes that require user input and confirmation! (i.e. training the model async, etc.)
- I forgot to set up checkpoints for the model. That's an easy fix I'll implement at some point.

I plan to revisit this project later on in my programming journey and see how I can improve it. It was a fun experience and taught me a lot about Flask web dev and handling backend development.

How to Run:
Download the repo, and run the main.py file. Once you have gotten to this point, and you have the application open, here is the way to set up your account and test some transactions! (though it is pretty self explanatory...)

![image](https://user-images.githubusercontent.com/77950550/197429316-53171bd7-db68-4d32-bc8a-d78cf28d4e9d.png)

If you have a pre-existing account (which you won't if this is your first time using it), click sign-up and create an account! It will prompt you for according details and then will request images of you to store in the database! This is because a biometric transaction platform needs your biometric data to ensure it's you at the time of approval of transaction. 

Once that's done, you'll be forwarded to the user menu, right? No...you'll instead be forwarded to a loading screen. You'll be here a while. This is where the model is trained on the data you just gave it. It says it on the page as well, but don't shut down the program or reload the page at this point - it's a very long process to train the model and interfering with the page will result in the model halting its training. So...don't do that.

At this point, when it finishes, it'll forward you to the user menu, which will give you your list of incoming and outgoing transactions. If you would like to, set up another test account and send transactions to that test account, and have that test account send some to you! It's kind of fun...maybe...not really.

If you already did this process and have your account, you can just press the login info, put in your basic information (no biometrics needed for simple auth), and then you're good to go!

The idea of this program is to be a proof of concept of a user oriented biometrics security platform, and while it definitely could use a LOT of improvement, it stands as a good living example of what I had envisioned at the beginning of the project.

# The architecture of the Siamese Model is here:

Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_img (InputLayer)          [(None, 105, 105, 3) 0

valid_img (InputLayer)          [(None, 105, 105, 3) 0

conv2d (Conv2D)                 (None, 96, 96, 64)   19264       input_img[0][0]

conv2d_4 (Conv2D)               (None, 96, 96, 64)   19264       valid_img[0][0]

max_pooling2d (MaxPooling2D)    (None, 48, 48, 64)   0           conv2d[0][0]

max_pooling2d_3 (MaxPooling2D)  (None, 48, 48, 64)   0           conv2d_4[0][0]

conv2d_1 (Conv2D)               (None, 42, 42, 128)  401536      max_pooling2d[0][0]

conv2d_5 (Conv2D)               (None, 42, 42, 128)  401536      max_pooling2d_3[0][0]

max_pooling2d_1 (MaxPooling2D)  (None, 21, 21, 128)  0           conv2d_1[0][0]

max_pooling2d_4 (MaxPooling2D)  (None, 21, 21, 128)  0           conv2d_5[0][0]

conv2d_2 (Conv2D)               (None, 18, 18, 128)  262272      max_pooling2d_1[0][0]

conv2d_6 (Conv2D)               (None, 18, 18, 128)  262272      max_pooling2d_4[0][0]

max_pooling2d_2 (MaxPooling2D)  (None, 9, 9, 128)    0           conv2d_2[0][0]

max_pooling2d_5 (MaxPooling2D)  (None, 9, 9, 128)    0           conv2d_6[0][0]

conv2d_3 (Conv2D)               (None, 6, 6, 256)    524544      max_pooling2d_2[0][0]

conv2d_7 (Conv2D)               (None, 6, 6, 256)    524544      max_pooling2d_5[0][0]

flatten (Flatten)               (None, 9216)         0           conv2d_3[0][0]

flatten_1 (Flatten)             (None, 9216)         0           conv2d_7[0][0]

dense (Dense)                   (None, 4096)         37752832    flatten[0][0]

dense_1 (Dense)                 (None, 4096)         37752832    flatten_1[0][0]

tf.math.subtract (TFOpLambda)   (None, 4096)         0           dense[0][0]
                                                                 dense_1[0][0]

tf.math.abs (TFOpLambda)        (None, 4096)         0           tf.math.subtract[0][0]

dense_2 (Dense)                 (None, 1)            4097        tf.math.abs[0][0]

Total params: 77,924,993
Trainable params: 77,924,993
Non-trainable params: 0
__________________________________________________________________________________________________
