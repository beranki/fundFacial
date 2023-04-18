# fundFacial
![image](https://user-images.githubusercontent.com/77950550/197428516-190b99f0-c579-444e-b25d-dace67f82509.png)

This project should be used as a marker for my journey into web development. 

## Cool Things:
- I felt I made an easy to use and innovative interface to handle the transaction requests themselves, and it was definitely a leap out of my comfort zone as I utilized schemas and other essentials to UX dev which I had not prior.

I plan to revisit this project later on in my programming journey and see how I can improve it. It was a fun experience and taught me a lot about Flask web dev and handling backend development.

## How to Run:
Download the repo, and run the main.py file. Once you have gotten to this point, and you have the application open, here is the way to set up your account and test some transactions! (though it is pretty self explanatory...)

![image](https://user-images.githubusercontent.com/77950550/197429316-53171bd7-db68-4d32-bc8a-d78cf28d4e9d.png)

If you have a pre-existing account (which you won't if this is your first time using it), click sign-up and create an account! It will prompt you for according details and then will request images of you to store in the database! This is because a biometric transaction platform needs your biometric data to ensure it's you at the time of approval of transaction. 

Once that's done, you'll be forwarded to the user menu, right? No...you'll instead be forwarded to a loading screen. You'll be here a while. This is where the model is trained on the data you just gave it. It says it on the page as well, but don't shut down the program or reload the page at this point - it's a very long process to train the model and interfering with the page will result in the model halting its training. 

At this point, when it finishes, it'll forward you to the user menu, which will give you your list of incoming and outgoing transactions. If you would like to, set up another test account and send transactions to that test account, and have that test account send some to you! It's kind of fun...maybe...not really.

If you already did this process and have your account, you can just press the login button on the navigation bar, put in your basic information (no biometrics needed for login auth), and then you're good to go!

The idea of this program is to be a proof of concept of a user oriented biometrics security platform, and while it definitely could use some improvement, it stands as a good living example of what I had envisioned at the beginning of the project.

## Siamese Model Architecture:
- implemented using the following paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- image taken directly from paper, as was implemented identically in project itself.
![image](https://user-images.githubusercontent.com/77950550/200202133-c5d0b409-1b8b-4ca2-9dc8-9462c65e003e.png)

## Things to Improve: 
- Definitely switching to a transfer model for face recognition...big big mistake to custom train it per user.
- Improve security and continue working towards making it a more user friendly app.
- Automate a lot of the processes that require user input and confirmation! (i.e. training the model async, etc.)
- I forgot to set up checkpoints for the model. That's an easy fix I'll implement at some point.
