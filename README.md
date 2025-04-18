# Motion-Detection-API
Several functions for video Motion Detection built in Python using FastAPI, OpenCV, and Streamlit

Tasks:

Face Detection (2)
- dlib
- haar cascade

Object motion detection (2)
- Color tracking
- ORB tracking

Motion detection (4)
- Lucas Kanade Sparse OF
- Farneback Dense OF
- Background Subtraction
- Frame Differencing

Instructions to run:
- pip install -r requirements.txt
- uvicorn api:app --reload --port=8080
- streamlit run frontend.py

![Frontend](https://cdn.discordapp.com/attachments/1090109948383481876/1362589453356371998/image.png?ex=6802f1d1&is=6801a051&hm=8967fae9409181a66e7b0c24abfd69ba5a3d7e37a9c0ebb2d068fe50cf61a3c9)
![Demo](https://cdn.discordapp.com/attachments/1090109948383481876/1362596385551159437/ezgif-607031ad90fd08.gif?ex=6802f846&is=6801a6c6&hm=f65f6031403113196b983488ddb4ca15c073587e7d2e0659bbd4e05f690ddbd5)

