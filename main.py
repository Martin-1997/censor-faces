import cv2
import mediapipe as mp
import os
import click
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm

def process_img_mediapipe(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box #relative_bounding_box

            # These values are relative
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W) if int(x1 * W) > 0 else 0 # It seems to happen that a negative value is returned, we use 0 instead
            y1 = int(y1 * H) if int(y1 * H) > 0 else 0 # It seems to happen that a negative value is returned, we use 0 instead
            w = int(w * W)
            h = int(h * H)

            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)
            # blur faces
            area_to_blur = img[y1:y1 + h, x1:x1 + w, :]
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(area_to_blur, (150,150))
    return img

def process_image_cv(img, face_frontal_classifier, face_profile_classifier):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_frontal = face_frontal_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))

    face_profile= face_profile_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))

    for (x, y, w, h) in face_frontal:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        area_to_blur = img[y:y + h, x:x + w, :]
        img[y:y + h, x:x + w, :] = cv2.blur(area_to_blur, (150,150))

    for (x, y, w, h) in face_profile:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        area_to_blur = img[y:y + h, x:x + w, :]
        img[y:y + h, x:x + w, :] = cv2.blur(area_to_blur, (150,150))

    return img

def process_image_retina_face(img, draw_box = False, box_scale = 1.1, blurring_intensity = 150):
    faces = RetinaFace.detect_faces(img)
    for face_key in faces:
        # print(face_key)
        # print(faces)
        (x1, y1, x2, y2) = faces[face_key]['facial_area']
        w = x2 - x1
        h = y2 - y1
        scaled_w = w * (box_scale - 1)
        scaled_h = h * (box_scale - 1)
        temp_x1 = int(x1 - scaled_w / 2)
        x1 = temp_x1 if temp_x1 > 0 else 0
        temp_y1 = int(y1 - scaled_h / 2)
        y1 = temp_y1 if temp_y1 > 0 else 0
        temp_x2 = int(x2 + scaled_w / 2)
        x2 = temp_x2 if temp_x2 > 0 else 0
        temp_y2 = int(y2 + scaled_h / 2)
        y2 = temp_y2 if temp_y2 > 0 else 0
        if draw_box:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        area_to_blur = img[y1:y2, x1:x2, :]
        img[y1:y2, x1:x2, :] = cv2.blur(area_to_blur, (blurring_intensity,blurring_intensity))
        # example content of faces[face_key]
        # {'score': 0.997837245464325, 'facial_area': [329, 881, 369, 937], 'landmarks': {'right_eye': [348.06262, 906.4378], 'left_eye': [362.34924, 905.0102], 'nose': [360.60382, 913.3919], 'mouth_right': [351.92596, 924.2114], 'mouth_left': [362.8363, 923.4533]}}
    return img

def check_and_create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            print(f"{dir} was created because it was missing.")
            os.makedirs(dir)

@click.command()
@click.option('--input-dir', type=str, default='input')
@click.option('--output-dir', type=str, default='output')
@click.option('--mode', default='video', type=click.Choice(['image', 'video'], case_sensitive=False))
def main(input_dir, output_dir, mode):
    output_dir
    check_and_create_dirs([input_dir, output_dir])
    files = [f for f in os.listdir(input_dir) if (os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] == 'mp4')]
    if len(files) < 1:
        print("There are no .mp4-files in the input dir")
        return 0

    for file_idx in range(len(files)):
        print(f"Processing file {file_idx}/{len(files)}: {files[file_idx]}")
        # mediapipe
        # mp_face_detection = mp.solutions.face_detection
        #model_selection=0 -> faces < 2m of camera
        #model_selection=1 -> faces < 5m of camera
        # with mp_face_detection.FaceDetection(min_detection_confidence=0.01, model_selection=1) as face_detection:

        # opencv
        # face_frontal_classifier = cv2.CascadeClassifier(
        # cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # face_profile_classifier = cv2.CascadeClassifier(
        # cv2.data.haarcascades + "haarcascade_profileface.xml")

        input_file = os.path.join(input_dir, files[file_idx])
        (file_name, file_extension) = files[file_idx].split(".")
        output_file = os.path.join(output_dir, file_name + "_censored.mp4")

        if mode == 'image':
            # read image
            img = cv2.imread(input_file)
            img = process_image_retina_face(img)
            # save image
            cv2.imwrite(os.path.join(output_dir, output_file), img)

        elif mode == 'video':
            cap = cv2.VideoCapture(input_file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = cap.read()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video = cv2.VideoWriter(output_file,
                                            fourcc,
                                            30.0, #fps
                                            (w, h))
            for i in tqdm(range(frame_count)):
            # while ret:
                frame_copy = frame
                try:
                    # frame = process_img(frame, face_detection)
                    # frame = process_image_cv(frame, face_frontal_classifier, face_profile_classifier)
                    frame = process_image_retina_face(frame, draw_box = False, box_scale = 1.5, blurring_intensity = 150)
                except TypeError:
                    frame = frame_copy
                finally:
                    output_video.write(frame)
                    ret, frame = cap.read()
                    #cv2.imshow('frame', frame)
                    # if cv2.waitKey(1) == ord('q'):
                    #     break

            # Release everything if job is finished
            cap.release()
            output_video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
