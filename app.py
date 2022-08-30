import face_recognition as fr
from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import csv
import pandas as pd

# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()

parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int,
                    help="ID of the camera. An integer between 0 and N. Default is 1")
parser.add_argument("-k", "--known", dest="known", default="known_people", type=str,
                    help="folder with the known people")
parser.add_argument("-r", "--rescale", dest="scaleDown", default=0.25, type=float,
                    help="scaling factor: 0.25 means 25 percent of the original size.")
parser.add_argument("-a", "--avatar_dimension", dest="avatarDim", default=100, type=int,
                    help="scaling factor: 0.25 means 25 percent of the original size.")

args = parser.parse_args()
# -------------------------------------------------- #

# - LAYOUT PARAMETERS ------------------------------ #
thickness = 2
color = (0, 0, 0)
avatar_w_dimension = args.avatarDim
tolerance = 5
distance = 30
frame_border = 0.2 # from 0 to 1 (from 0% to 100%)

# -------------------------------------------------- #

def can_be_right(xf, frame_width, avatar_width, distance, tolerance):
    return xf + distance + avatar_width + tolerance < frame_width


def can_be_left(xi, avatar_width, distance, tolerance):
    return xi - distance - avatar_width - tolerance > 0


def can_be_top(yi, avatar_length, distance, tolerance):
    return yi - distance - avatar_length - tolerance > 0


def can_be_bottom(yf, frame_length, avatar_length, distance, tolerance):
    return yf + distance + avatar_length + tolerance < frame_length


current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

info_social = pd.read_csv(current_directory + '/' + 'info.csv')

# load known people
list_people_path = [f for f in listdir(current_directory + "/known_people/") if isfile(join(current_directory + '/' + args.known, f))]
list_avatar_path = [f for f in listdir(current_directory + "/known_avatar/") if isfile(join(current_directory + "/known_avatar", f))]
list_people_name = [f.split(".")[0] for f in listdir(current_directory + "/known_people/") if isfile(join(current_directory + '/' + args.known, f))]
list_avatar_name = [f.split(".")[0] for f in listdir(current_directory + "/known_avatar/") if isfile(join(current_directory + "/known_avatar", f))]
print(f'\n{len(list_people_path)} known people in your database found.')


difference_ka = list(set(list_people_path) - set(list_avatar_path))
difference_ak = list(set(list_avatar_path) - set(list_people_path))

if len(difference_ak) > 0:
    # remove old avatar
    pass

if len(difference_ka) > 0:
    print("\n-- Updating Database")
    print("\n1. Updating Avatar...")
    for face_file_name in difference_ka:
        absolute_path = os.path.join(os.getcwd(), args.known, face_file_name)
        person = cv2.imread(absolute_path)
        person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        face_det = fr.face_locations(person_rgb)[0]
        person_rgb = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)
        border_w = int((face_det[1]-face_det[3])*frame_border)
        border_h = int((face_det[2]-face_det[0])*frame_border)
        io = person_rgb[max(face_det[0]-border_h, 0):min(face_det[2]+border_h, person_rgb.shape[0]),
                        max(face_det[3]-border_w, 0):min(face_det[1]+border_w, person_rgb.shape[1])
                        ]
        cv2.imwrite(f'./known_avatar/{face_file_name}', io)
    print('DONE')

    print("\n2. Updating Model...")
    # extract feature from know people
    list_people_encoded = []
    for face_file_name in list_people_path:
        absolute_path = os.path.join(os.getcwd(), args.known, face_file_name)
        person = cv2.imread(absolute_path)
        person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        list_people_encoded.append(fr.face_encodings(person_rgb)[0])
    print('DONE')

    print("\n3. Saving New Model...")
    with open("model.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_people_encoded)
    print('DONE')

else:
    print("\n-- Loading Model")
    list_people_encoded = []

    with open(current_directory + '/model.csv', 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj)
        # convert string to list
        list_people_encoded = list(csv_reader)

    # float convertion
    list_people_encoded = [list(map(float, sublist)) for sublist in list_people_encoded]

    print('DONE')


cap = cv2.VideoCapture(args.camera)
while cap.isOpened():
    ret, frame = cap.read()
    frame_scaled = cv2.resize(frame, (0, 0), fx=args.scaleDown, fy=args.scaleDown)
    rgb_frame_scaled = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame_scaled)

    if len(face_locations) > 0:
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

            width = int((yf - yi) / 5)
            height = int((xf - xi) / 5)

            # if name != 'Unknown2':
            avatar = cv2.imread(f'{current_directory}/known_avatar/{name}.jpg')
            avatar = cv2.resize(avatar,
                                (int(avatar_w_dimension * avatar.shape[1] / avatar.shape[0]), avatar_w_dimension))

            frame_length = frame.shape[0]
            frame_width = frame.shape[1]
            avatar_length = avatar.shape[0]
            avatar_width = avatar.shape[1]

            right = can_be_right(xf, frame_width, avatar_width, distance, tolerance)
            left = can_be_left(xi, avatar_width, distance, tolerance)
            top = can_be_top(yi, avatar_length, distance, tolerance)
            bottom = can_be_bottom(yf, frame_length, avatar_length, distance, tolerance)

            # draw_avatar = False
            if top:
                if right:
                    avatar_yi = yi - distance - avatar_w_dimension
                    avatar_yf = yi - distance - avatar_w_dimension + avatar_length
                    avatar_xi = xf + distance
                    avatar_xf = xf + distance + avatar_width

                    line_xi = xf
                    line_xf = xf + distance
                    line_yi = yi
                    line_yf = yi - distance

                    draw_avatar = True
                elif left:
                    avatar_yi = yi - distance - avatar_w_dimension
                    avatar_yf = yi - distance - avatar_w_dimension + avatar_length
                    avatar_xi = xi - distance - avatar_w_dimension
                    avatar_xf = xi - distance - avatar_w_dimension + avatar_width

                    line_xi = xi
                    line_xf = xi - distance
                    line_yi = yi
                    line_yf = yi - distance

                    draw_avatar = True
                else:
                    print('to be implemented')
                    # put in the top middle
            elif bottom:
                if right:
                    avatar_yi = yf + distance
                    avatar_yf = yf + distance + avatar_length
                    avatar_xi = xf + distance
                    avatar_xf = xf + distance + avatar_width

                    line_xi = xf
                    line_xf = xf + distance
                    line_yi = yf
                    line_yf = yf + distance

                    draw_avatar = True
                elif left:
                    avatar_yi = yf + distance
                    avatar_yf = yf + distance + avatar_length
                    avatar_xi = xi - distance - avatar_w_dimension
                    avatar_xf = xi - distance - avatar_w_dimension + avatar_width

                    line_xi = xi
                    line_xf = xi - distance
                    line_yi = yf
                    line_yf = yf + distance

                    draw_avatar = True
                else:
                    print('to be implemented')
                    # put in the bottom middle
            else:
                print('nothing to do at the moment!')

            if draw_avatar:
                frame[avatar_yi:avatar_yf, avatar_xi:avatar_xf] = avatar

                cv2.line(frame, (line_xi, line_yi), (line_xf, line_yf), color, thickness)

                cv2.rectangle(frame, (avatar_xi, avatar_yi), (avatar_xf, avatar_yf), color,
                              thickness)

            cv2.putText(frame, name, (avatar_xf+10, avatar_yi + 20), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)
            cv2.putText(frame, f'Birth Date: {info_social[info_social["name"]==name].birthday.values[0]}', (avatar_xf+10, avatar_yi + 45), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
            cv2.putText(frame, f'Facebook: {info_social[info_social["name"]==name].fb.values[0]}', (avatar_xf+10, avatar_yi + 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
            cv2.putText(frame, f'Instagram: {info_social[info_social["name"]==name].ig.values[0]}', (avatar_xf+10, avatar_yi + 85), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)

            cv2.line(frame, (xi, yi), (xi + width, yi), color, thickness)
            cv2.line(frame, (xi, yi), (xi, yi + height), color, thickness)

            cv2.line(frame, (xf, yf), (xf - width, yf), color, thickness)
            cv2.line(frame, (xf, yf), (xf, yf - height), color, thickness)

            cv2.line(frame, (xi, yf), (xi + width, yf), color, thickness)
            cv2.line(frame, (xi, yf), (xi, yf - height), color, thickness)

            cv2.line(frame, (xf, yi), (xf - width, yi), color, thickness)
            cv2.line(frame, (xf, yi), (xf, yi + height), color, thickness)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
