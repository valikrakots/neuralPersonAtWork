import cv2

videopath = './video.mp4'
start_frame = 19*60*15
skip_frame = (6*60+52)*60*15

def obtain_data():
    cap = cv2.VideoCapture(videopath)
    print(cap.get(cv2.CAP_PROP_FPS))
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(amount_of_frames)
    frame_number = start_frame
    i = 1
    while frame_number <= skip_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = cap.read()
        if res:
            # cropped_image = frame[115:350, 210:445]
            # cv2.imwrite('unsorted_data/' + str(i) + '.jpg', cropped_image)
            cv2.imwrite('unsorted_data/' + str(i) + '.jpg', frame)
        frame_number += 15*30
        i += 1
    print('Data obtained')

obtain_data()
