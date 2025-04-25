import face_recognition as fr
from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import csv
import pandas as pd
import re
import hashlib

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
parser.add_argument("-f", "--force_update", dest="force_update", action="store_true",
                    help="Force model update even if no new people are detected")

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


def get_base_name(filename):
    # Extract base name from filename (removing _1, _2, etc.)
    match = re.match(r'(.+?)(?:_\d+)?\.[^.]+$', filename)
    if match:
        return match.group(1)
    else:
        return os.path.splitext(filename)[0]


def calculate_files_hash(directory):
    """Calculate a hash of all files in the directory to detect changes"""
    hash_obj = hashlib.md5()
    files = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
    for file_name in files:
        file_path = join(directory, file_name)
        file_stat = os.stat(file_path)
        file_info = f"{file_name}_{file_stat.st_size}_{file_stat.st_mtime}"
        hash_obj.update(file_info.encode())
    return hash_obj.hexdigest()

print('\n-- Checking files and model')
current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
print(f'\n [x] Current directory: {current_directory}')

# Check if the model needs to be updated by calculating a hash of the current files
known_dir = current_directory + "/" + args.known
print(f' [x] Known directory: {known_dir}')
last_hash_file = current_directory + "/last_hash.txt"
current_hash = calculate_files_hash(known_dir)
print(f' [x] Current hash: {current_hash}')
last_hash = ""

if os.path.exists(last_hash_file):
    with open(last_hash_file, 'r') as file:
        last_hash = file.read().strip()
        print(f' [x] Last hash: {last_hash}')

model_needs_update = (current_hash != last_hash) or args.force_update

# Load social info
info_social = pd.read_csv(current_directory + '/' + 'info.csv')


print('\n-- Checking Database')

# load known people
list_people_path = [f for f in listdir(known_dir) if isfile(join(known_dir, f))]
list_avatar_path = [f for f in listdir(current_directory + "/known_avatar/") if isfile(join(current_directory + "/known_avatar", f))]

# Map to store base names to all their image files
person_to_images = {}
for file_path in list_people_path:
    base_name = get_base_name(file_path)
    if base_name not in person_to_images:
        person_to_images[base_name] = []
    person_to_images[base_name].append(file_path)

# Sort the images for each person to ensure the main image (without suffix) comes first
for person in person_to_images:
    person_to_images[person].sort()

# Get list of unique people
list_unique_people = list(person_to_images.keys())

print(f'\n [x] {len(list_people_path)} photos of {len(list_unique_people)} known people in the database found.')

# Check for new people that need avatars (people who don't have an avatar yet)
new_people = []
for person in list_unique_people:
    avatar_file = f'{person}.jpg'
    if avatar_file not in list_avatar_path:
        new_people.append(person)

if len(new_people) > 0:
    print(f' [x] {len(new_people)} found')

if len(new_people) > 0 or model_needs_update:
    print("\n-- Updating Database & Model")

    # Create avatars for new people
    if len(new_people) > 0:
        print("\n [-] Creating New Avatars...")
        for person in new_people:
            # Use the first image of this person to create the avatar
            face_file_name = person_to_images[person][0]  # First image for this person
            avatar_output_name = f"{person}.jpg"

            absolute_path = os.path.join(os.getcwd(), args.known, face_file_name)
            person_img = cv2.imread(absolute_path)
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            faces = fr.face_locations(person_rgb)
            if len(faces) > 0:
                face_det = faces[0]
                person_rgb = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)
                border_w = int((face_det[1]-face_det[3])*frame_border)
                border_h = int((face_det[2]-face_det[0])*frame_border)
                io = person_rgb[max(face_det[0]-border_h, 0):min(face_det[2]+border_h, person_rgb.shape[0]),
                                max(face_det[3]-border_w, 0):min(face_det[1]+border_w, person_rgb.shape[1])
                               ]
                cv2.imwrite(f'./known_avatar/{avatar_output_name}', io)
                print(f"      -> Created avatar for {person}")
            else:
                print(f"Warning: No face detected in {face_file_name}, avatar not created for {person}")

    else:
        print("\n [-] No New Avatars")

    print("\n [-] Updating Model...")
    # extract feature from known people (all images)
    list_people_encoded = []
    list_people_names = []

    # Process all images for all people
    for person in list_unique_people:
        for face_file_name in person_to_images[person]:
            absolute_path = os.path.join(os.getcwd(), args.known, face_file_name)
            person_img = cv2.imread(absolute_path)
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            faces = fr.face_encodings(person_rgb)
            if len(faces) > 0:
                list_people_encoded.append(faces[0])
                list_people_names.append(person)  # Store the base name, not the filename
                #print(f"Added encoding for {face_file_name}")
            else:
                print(f" Warning: No face encoding could be generated for {face_file_name}")


    print("\n [-] Saving New Model...")
    # Save model and names
    with open("model.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(list_people_encoded)

    with open("names.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerow(list_people_names)

    # Save the current hash to avoid unnecessary updates
    with open(last_hash_file, 'w') as file:
        file.write(current_hash)


else:
    print("\n-- Loading Existing Model")
    list_people_encoded = []
    list_people_names = []

    print(f'\n [x] Loading model.csv')

    with open(current_directory + '/model.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        list_people_encoded = list(csv_reader)

    # float conversion
    list_people_encoded = [list(map(float, sublist)) for sublist in list_people_encoded]

    # Load names
    try:
        print(f' [x] Loading names.csv')

        with open(current_directory + '/names.csv', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            list_people_names = next(csv_reader)  # Read the single row of names
    except FileNotFoundError:
        # For backward compatibility with older versions
        list_people_names = [get_base_name(f) for f in list_people_path]
        print("Warning: names.csv not found, using filenames as person identifiers")

print("\n-- Starting recognition")


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
                name = list_people_names[best_match_index]
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

            if name == "Unknown":
                # Use a default unknown avatar
                avatar_path = f'{current_directory}/known_avatar/Unknown.jpg'
            else:
                # Use the person's avatar
                avatar_file = f'{name}.jpg'
                avatar_path = f'{current_directory}/known_avatar/{avatar_file}'

            if os.path.exists(avatar_path):
                avatar = cv2.imread(avatar_path)
                avatar = cv2.resize(avatar,
                      (int(avatar_w_dimension * avatar.shape[1] / avatar.shape[0]), avatar_w_dimension))
                # Use base name for avatar (no suffix)
                avatar_file = f'{name}.jpg'
                avatar_path = f'{current_directory}/known_avatar/{avatar_file}'

                if os.path.exists(avatar_path):
                    avatar = cv2.imread(avatar_path)
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

                    draw_avatar = False
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

                        cv2.rectangle(frame, (avatar_xi, avatar_yi), (avatar_xf, avatar_yf), color, thickness)

                        # Check if the person exists in info_social DataFrame before displaying info
                        if name in info_social['name'].values:
                            cv2.putText(frame, name, (avatar_xf+10, avatar_yi + 20), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)
                            cv2.putText(frame, f'Birth Date: {info_social[info_social["name"]==name].birthday.values[0]}', (avatar_xf+10, avatar_yi + 45), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(frame, f'Facebook: {info_social[info_social["name"]==name].fb.values[0]}', (avatar_xf+10, avatar_yi + 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                            cv2.putText(frame, f'Instagram: {info_social[info_social["name"]==name].ig.values[0]}', (avatar_xf+10, avatar_yi + 85), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                        else:
                            cv2.putText(frame, name, (avatar_xf+10, avatar_yi + 20), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)
                else:
                    print(f"Avatar not found for {name}")

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
