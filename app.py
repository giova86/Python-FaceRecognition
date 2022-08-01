import face_recognition as fr
from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np
from argparse import ArgumentParser

# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()

parser.add_argument("-c", "--camera_id", dest="camera", default=1, type=int,
                    help="ID of the camera. An integer between 0 and N. Default is 1")
parser.add_argument("-k", "--known", dest="known", default="known_people", type=str,
                    help="folder with the known people")
parser.add_argument("-r", "--scale", dest="scaleDown", default=0.25, type=float,
                    help="scaling factor: 0.25 means 25% of the original size.")

args = parser.parse_args()
# -------------------------------------------------- #

# - LAYOUT PARAMETERS ------------------------------ #
thickness = 4
color = (255,255,255)
avatar_w_dimension = 100
# -------------------------------------------------- #

# load known people
list_people_path = [join(args.known, f) for f in listdir("known_people/") if isfile(join(args.known, f))]
list_people_name = [f.split(".")[0] for f in listdir("known_people/") if isfile(join(args.known, f))]
print(f'{len(list_people_path)} known people in your database found.')

# extract feature from know people
list_people_encoded = []
print('Upgrading Database...')
for i in list_people_path:
    person = cv2.imread(i)
    person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    list_people_encoded.append(fr.face_encodings(person_rgb)[0])

    if os.path.isfile(f'./known_avatar/{i.split("/")[1]}'):
        pass
    else:
        face_det = fr.face_locations(person_rgb)[0]
        person_rgb = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)
        io = person_rgb[face_det[0]:face_det[2], face_det[3]:face_det[1]]
        cv2.imwrite(f'./known_avatar/{i.split("/")[1]}', io)
print('Done')

cap = cv2.VideoCapture(args.camera)
while cap.isOpened():
    ret, frame = cap.read()
    frame_scaled = cv2.resize(frame, (0, 0), fx=args.scaleDown, fy=args.scaleDown)
    rgb_frame_scaled = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame_scaled)
    face_encodings = fr.face_encodings(rgb_frame_scaled, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = fr.compare_faces(list_people_encoded, face_encoding)

        face_distances = fr.face_distance(list_people_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = list_people_name[best_match_index]
            face_names.append(name)
        else:
            name = "Unknown"
            face_names.append(name)

    face_locations = np.array(face_locations)
    face_locations = (face_locations / args.scaleDown).astype(int)

    for face_coordinates, name in zip(face_locations, face_names):
        yi, xf, yf, xi = face_coordinates[0], face_coordinates[1], face_coordinates[2], face_coordinates[3]

        width = int((yf - yi)/5)
        height = int((xf - xi)/5)

        if name != 'Unknown':
            avatar = cv2.imread(f'./known_avatar/{name}.jpg')
            avatar = cv2.resize(avatar, (int(avatar_w_dimension*avatar.shape[1]/avatar.shape[0]),avatar_w_dimension) )
            #avatar = cv2.resize(avatar, (0, 0), fx=args.scaleDown, fy=args.scaleDown)
            avatar_shape = avatar.shape
            frame[(yi-30-avatar_w_dimension):(yi-30-avatar_w_dimension+avatar_shape[0]), (xf+30):(xf+30 + avatar_shape[1])] = avatar
            cv2.line(frame, (xf,yi),(xf+30,yi-30), color, thickness)
            cv2.rectangle(frame, (xf+30,yi-30),(xf+30+avatar_shape[1],yi-30-avatar_shape[0]), color, thickness)

        cv2.putText(frame, name, (xi, yi-20), cv2.FONT_HERSHEY_DUPLEX,1, (0,200,0), 2)

        cv2.line(frame, (xi,yi),(xi+width,yi), color, thickness)
        cv2.line(frame, (xi,yi),(xi,yi+height), color, thickness)

        cv2.line(frame, (xf,yf),(xf-width,yf), color, thickness)
        cv2.line(frame, (xf,yf),(xf,yf-height), color, thickness)

        cv2.line(frame, (xi,yf),(xi+width,yf), color, thickness)
        cv2.line(frame, (xi,yf),(xi,yf-height), color, thickness)

        cv2.line(frame, (xf,yi),(xf-width,yi), color, thickness)
        cv2.line(frame, (xf,yi),(xf,yi+height), color, thickness)


    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
