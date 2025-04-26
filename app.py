import face_recognition as fr
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import csv
import pandas as pd
import re
import hashlib
import time
import argparse

def pprint(text, char_delay=0.003, line_delay=0.003):
    """Pretty print with typing effect"""
    lines = text.split('\n')
    for line in lines:
        for char in line:
            print(char, end='', flush=True)
            time.sleep(char_delay)
        print()  # New line after each line
        time.sleep(line_delay)  # Pause after each line

def get_base_name(filename):
    """Extract base name from filename (removing _1, _2, etc.)"""
    match = re.match(r'(.+?)(?:_\d+)?\.[^.]+$', filename)
    if match:
        return match.group(1)
    else:
        return os.path.splitext(filename)[0]

def calculate_files_hash(directory):
    """Calculate a hash of all files in the directory to detect changes"""
    hash_obj = hashlib.md5()
    if not os.path.exists(directory):
        return "directory_not_found"

    files = sorted([f for f in listdir(directory) if isfile(join(directory, f)) and f != '.DS_Store'])
    for file_name in files:
        file_path = join(directory, file_name)
        file_stat = os.stat(file_path)
        file_info = f"{file_name}_{file_stat.st_size}_{file_stat.st_mtime}"
        hash_obj.update(file_info.encode())
    return hash_obj.hexdigest()

def can_be_right(xf, frame_width, avatar_width, distance, tolerance):
    return xf + distance + avatar_width + tolerance < frame_width

def can_be_left(xi, avatar_width, distance, tolerance):
    return xi - distance - avatar_width - tolerance > 0

def can_be_top(yi, avatar_length, distance, tolerance):
    return yi - distance - avatar_length - tolerance > 0

def can_be_bottom(yf, frame_length, avatar_length, distance, tolerance):
    return yf + distance + avatar_length + tolerance < frame_length

def is_valid_image_file(file_path):
    """Check if a file is a valid image file and not a system file like .DS_Store"""
    # Skip .DS_Store and other hidden files
    if file_path.startswith('.'):
        return False

    # Check common image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    file_ext = os.path.splitext(file_path)[1].lower()

    return file_ext in valid_extensions

def main():
    # Command line arguments setup
    parser = argparse.ArgumentParser(description='Face Recognition Application')
    parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int,
                        help="ID of the camera. An integer between 0 and N. Default is 0")
    parser.add_argument("-k", "--known", dest="known", default="known_people", type=str,
                        help="folder with the known people")
    parser.add_argument("-r", "--rescale", dest="scaleDown", default=0.25, type=float,
                        help="scaling factor: 0.25 means 25 percent of the original size.")
    parser.add_argument("-a", "--avatar_dimension", dest="avatarDim", default=100, type=int,
                        help="dimension for the avatar images")
    parser.add_argument("-f", "--force_update", dest="force_update", action="store_true",
                        help="Force model update even if no new people are detected")
    args = parser.parse_args()

    # Layout parameters
    thickness = 2
    color = (0, 0, 0)
    avatar_w_dimension = args.avatarDim
    tolerance = 5
    distance = 30
    frame_border = 0.2  # from 0 to 1 (from 0% to 100%)

    pprint('\n-- Checking files and model')
    current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
    pprint(f'\n [x] Current directory: {current_directory}')

    # Ensure directories exist
    known_dir = os.path.join(current_directory, args.known)
    avatar_dir = os.path.join(current_directory, "known_avatar")

    # Create directories if they don't exist
    if not os.path.exists(known_dir):
        os.makedirs(known_dir)
        pprint(f' [x] Created known directory: {known_dir}')

    if not os.path.exists(avatar_dir):
        os.makedirs(avatar_dir)
        pprint(f' [x] Created avatar directory: {avatar_dir}')

        # Create a default unknown avatar
        unknown_avatar_path = os.path.join(avatar_dir, "Unknown.jpg")
        if not os.path.exists(unknown_avatar_path):
            # Create a simple blank image for unknown people
            unknown_img = np.ones((avatar_w_dimension, avatar_w_dimension, 3), dtype=np.uint8) * 200
            cv2.putText(unknown_img, "?", (int(avatar_w_dimension/3), int(avatar_w_dimension/1.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
            cv2.imwrite(unknown_avatar_path, unknown_img)
            pprint(f' [x] Created default unknown avatar')

    # Check if info.csv exists, if not create a template
    info_csv_path = os.path.join(current_directory, 'info.csv')
    if not os.path.exists(info_csv_path):
        with open(info_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'birthday', 'fb', 'ig'])
            writer.writerow(['Example_Person', '01/01/1990', 'example.user', '@example'])
        pprint(f' [x] Created template info.csv file')

    # Check if the model needs to be updated by calculating a hash of the current files
    pprint(f' [x] Known directory: {known_dir}')
    last_hash_file = os.path.join(current_directory, "last_hash.txt")
    current_hash = calculate_files_hash(known_dir)
    pprint(f' [x] Current hash: {current_hash}')
    last_hash = ""

    if os.path.exists(last_hash_file):
        with open(last_hash_file, 'r') as file:
            last_hash = file.read().strip()
            pprint(f' [x] Last hash: {last_hash}')

    model_needs_update = (current_hash != last_hash) or args.force_update

    # Load social info if available
    try:
        info_social = pd.read_csv(info_csv_path)
        pprint(f' [x] Loaded info.csv with {len(info_social)} entries')
    except Exception as e:
        pprint(f' [!] Error loading info.csv: {e}')
        info_social = pd.DataFrame(columns=['name', 'birthday', 'fb', 'ig'])

    pprint('\n-- Checking Database')

    # Load known people (filter out .DS_Store and other hidden files)
    if os.path.exists(known_dir):
        list_people_path = [f for f in listdir(known_dir) if isfile(join(known_dir, f)) and is_valid_image_file(f)]
        pprint(f'\n [x] Found {len(list_people_path)} valid image files')
    else:
        pprint(f'\n [!] Known directory {known_dir} does not exist')
        list_people_path = []

    if os.path.exists(avatar_dir):
        list_avatar_path = [f for f in listdir(avatar_dir) if isfile(join(avatar_dir, f)) and is_valid_image_file(f)]
    else:
        pprint(f' [!] Avatar directory {avatar_dir} does not exist')
        list_avatar_path = []

    # Map to store base names to all their image files
    person_to_images = {}
    for file_path in list_people_path:
        # Skip .DS_Store files
        if file_path == '.DS_Store':
            continue

        base_name = get_base_name(file_path)
        if base_name not in person_to_images:
            person_to_images[base_name] = []
        person_to_images[base_name].append(file_path)

    # Sort the images for each person to ensure the main image (without suffix) comes first
    for person in person_to_images:
        person_to_images[person].sort()

    # Get list of unique people
    list_unique_people = list(person_to_images.keys())

    pprint(f' [x] {len(list_people_path)} photos of {len(list_unique_people)} known people in the database found.')

    # Check for new people that need avatars (people who don't have an avatar yet)
    new_people = []
    for person in list_unique_people:
        avatar_file = f'{person}.jpg'
        if avatar_file not in list_avatar_path:
            new_people.append(person)

    if len(new_people) > 0:
        pprint(f' [x] {len(new_people)} new people found who need avatars')

    # NEW CODE: Find avatars that no longer have a corresponding person in known_people
    if os.path.exists(avatar_dir):
        # Get all avatar filenames except Unknown.jpg (we always want to keep this)
        existing_avatars = [f for f in list_avatar_path if f != "Unknown.jpg"]

        # Extract person names from avatar filenames (remove .jpg extension)
        existing_avatar_people = [os.path.splitext(f)[0] for f in existing_avatars]

        # Find people that have been removed (have avatar but no longer in known_people)
        removed_people = [person for person in existing_avatar_people if person not in list_unique_people]

        if removed_people:
            pprint(f' [x] Found {len(removed_people)} avatars of people who have been removed from the database')

    if len(new_people) > 0 or model_needs_update:
        pprint("\n-- Updating Database & Model")

        # NEW CODE: Delete avatars for removed people
        if model_needs_update and 'removed_people' in locals() and removed_people:
            pprint("\n [-] Removing obsolete avatars...")
            for person in removed_people:
                avatar_file = f'{person}.jpg'
                avatar_path = os.path.join(avatar_dir, avatar_file)
                if os.path.exists(avatar_path):
                    try:
                        os.remove(avatar_path)
                        pprint(f"      -> Removed avatar for {person}")
                    except Exception as e:
                        pprint(f"      Error removing avatar for {person}: {e}")

        # Create avatars for new people
        if len(new_people) > 0:
            pprint("\n [-] Creating New Avatars...")
            for person in new_people:
                # Use the first image of this person to create the avatar
                if person in person_to_images and person_to_images[person]:
                    face_file_name = person_to_images[person][0]  # First image for this person

                    # Skip .DS_Store files explicitly
                    if face_file_name == '.DS_Store':
                        pprint(f"      Skipping system file: {face_file_name}")
                        continue

                    avatar_output_name = f"{person}.jpg"

                    absolute_path = os.path.join(known_dir, face_file_name)

                    # Debug info
                    # pprint(f"      -> Processing image: {face_file_name}")

                    # Check if file exists
                    if not os.path.isfile(absolute_path):
                        pprint(f"      Error: File does not exist: {absolute_path}")
                        continue

                    # Try to read the image
                    person_img = cv2.imread(absolute_path)
                    if person_img is None:
                        pprint(f"      Error: Could not read image: {absolute_path}")
                        continue

                    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                    faces = fr.face_locations(person_rgb)
                    if len(faces) > 0:
                        face_det = faces[0]
                        person_rgb = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)
                        border_w = int((face_det[1]-face_det[3])*frame_border)
                        border_h = int((face_det[2]-face_det[0])*frame_border)

                        # Make sure indices are within image boundaries
                        top = max(face_det[0]-border_h, 0)
                        bottom = min(face_det[2]+border_h, person_rgb.shape[0])
                        left = max(face_det[3]-border_w, 0)
                        right = min(face_det[1]+border_w, person_rgb.shape[1])

                        io = person_rgb[top:bottom, left:right]

                        avatar_path = os.path.join(avatar_dir, avatar_output_name)
                        cv2.imwrite(avatar_path, io)
                        pprint(f"      -> Created avatar for {person}")
                    else:
                        pprint(f"      Warning: No face detected in {face_file_name}, avatar not created for {person}")
                else:
                    pprint(f"      Error: No images found for {person}")

        else:
            pprint("\n [-] No New Avatars Needed")

        pprint("\n [-] Updating Model...")
        # extract feature from known people (all images)
        list_people_encoded = []
        list_people_names = []

        # Process all images for all people
        for person in list_unique_people:
            for face_file_name in person_to_images[person]:
                # Skip .DS_Store files explicitly
                if face_file_name == '.DS_Store':
                    continue

                absolute_path = os.path.join(known_dir, face_file_name)

                # Check if file exists
                if not os.path.isfile(absolute_path):
                    pprint(f" Warning: File does not exist: {absolute_path}")
                    continue

                # Try to read the image
                person_img = cv2.imread(absolute_path)
                if person_img is None:
                    pprint(f" Warning: Could not read image: {absolute_path}")
                    continue

                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                faces = fr.face_encodings(person_rgb)
                if len(faces) > 0:
                    list_people_encoded.append(faces[0])
                    list_people_names.append(person)  # Store the base name, not the filename
                else:
                    pprint(f" Warning: No face encoding could be generated for {face_file_name}")

        # Save model and names if we have any face encodings
        if list_people_encoded:
            pprint("\n [-] Saving New Model...")
            model_path = os.path.join(current_directory, "model.csv")
            names_path = os.path.join(current_directory, "names.csv")

            # Save model and names
            with open(model_path, "w", newline='') as f:
                wr = csv.writer(f)
                wr.writerows(list_people_encoded)

            with open(names_path, "w", newline='') as f:
                wr = csv.writer(f)
                wr.writerow(list_people_names)

            # Save the current hash to avoid unnecessary updates
            with open(last_hash_file, 'w') as file:
                file.write(current_hash)
        else:
            pprint("\n [-] No face encodings found, model not updated")

    else:
        pprint("\n-- Loading Existing Model")
        list_people_encoded = []
        list_people_names = []

        model_path = os.path.join(current_directory, "model.csv")
        names_path = os.path.join(current_directory, "names.csv")

        if os.path.exists(model_path):
            pprint(f'\n [x] Loading model.csv')
            try:
                with open(model_path, 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    list_people_encoded = list(csv_reader)

                # float conversion
                list_people_encoded = [list(map(float, sublist)) for sublist in list_people_encoded]
            except Exception as e:
                pprint(f' [!] Error loading model.csv: {e}')
                list_people_encoded = []
        else:
            pprint(f' [!] model.csv not found')

        # Load names
        if os.path.exists(names_path):
            try:
                pprint(f' [x] Loading names.csv')
                with open(names_path, 'r') as read_obj:
                    csv_reader = csv.reader(read_obj)
                    list_people_names = next(csv_reader)  # Read the single row of names
            except Exception as e:
                pprint(f' [!] Error loading names.csv: {e}')
                # For backward compatibility with older versions
                list_people_names = [get_base_name(f) for f in list_people_path if f != '.DS_Store']
        else:
            pprint(f' [!] names.csv not found, using filenames as person identifiers')
            list_people_names = [get_base_name(f) for f in list_people_path if f != '.DS_Store']

    # Check if we have a valid model before proceeding
    if not list_people_encoded or not list_people_names:
        pprint("\n [!] No valid face recognition model found. Please add images to the known_people directory.")
        return

    pprint("\n-- Starting recognition")

    # Try to open the camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        pprint(f"\n [!] Error: Could not open camera {args.camera}")
        return

    pprint(f"\n [x] Camera opened successfully")
    pprint(f" [x] Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            pprint("\n [!] Error: Could not read frame from camera")
            break

        # Scale down frame for faster processing
        frame_scaled = cv2.resize(frame, (0, 0), fx=args.scaleDown, fy=args.scaleDown)
        rgb_frame_scaled = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame
        face_locations = fr.face_locations(rgb_frame_scaled)

        if len(face_locations) > 0:
            # Get face encodings for each face location
            face_encodings = fr.face_encodings(rgb_frame_scaled, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = fr.compare_faces(list_people_encoded, face_encoding)

                # Get distances to all known faces
                face_distances = fr.face_distance(list_people_encoded, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = list_people_names[best_match_index]
                    face_names.append(name)
                else:
                    name = "Unknown"
                    face_names.append(name)

            # Scale face locations back to original frame size
            face_locations = np.array(face_locations)
            face_locations = (face_locations / args.scaleDown).astype(int)

            # Draw rectangles and avatars for each face
            for face_coordinates, name in zip(face_locations, face_names):
                yi, xf, yf, xi = face_coordinates[0], face_coordinates[1], face_coordinates[2], face_coordinates[3]

                # Calculate dimensions for corner markers
                width = int((yf - yi) / 5)
                height = int((xf - xi) / 5)

                # Draw corner markers
                cv2.line(frame, (xi, yi), (xi + width, yi), color, thickness)
                cv2.line(frame, (xi, yi), (xi, yi + height), color, thickness)

                cv2.line(frame, (xf, yf), (xf - width, yf), color, thickness)
                cv2.line(frame, (xf, yf), (xf, yf - height), color, thickness)

                cv2.line(frame, (xi, yf), (xi + width, yf), color, thickness)
                cv2.line(frame, (xi, yf), (xi, yf - height), color, thickness)

                cv2.line(frame, (xf, yi), (xf - width, yi), color, thickness)
                cv2.line(frame, (xf, yi), (xf, yi + height), color, thickness)

                # Get avatar path
                if name == "Unknown":
                    # Use a default unknown avatar
                    avatar_path = os.path.join(avatar_dir, 'Unknown.jpg')
                else:
                    # Use the person's avatar
                    avatar_file = f'{name}.jpg'
                    avatar_path = os.path.join(avatar_dir, avatar_file)

                # Load and display avatar if it exists
                if os.path.exists(avatar_path):
                    avatar = cv2.imread(avatar_path)
                    if avatar is not None:
                        avatar = cv2.resize(avatar,
                            (int(avatar_w_dimension * avatar.shape[1] / avatar.shape[0]), avatar_w_dimension))

                        frame_length = frame.shape[0]
                        frame_width = frame.shape[1]
                        avatar_length = avatar.shape[0]
                        avatar_width = avatar.shape[1]

                        # Check available spaces for avatar placement
                        right = can_be_right(xf, frame_width, avatar_width, distance, tolerance)
                        left = can_be_left(xi, avatar_width, distance, tolerance)
                        top = can_be_top(yi, avatar_length, distance, tolerance)
                        bottom = can_be_bottom(yf, frame_length, avatar_length, distance, tolerance)

                        draw_avatar = False
                        if top:
                            if right:
                                avatar_yi = yi - distance - avatar_length
                                avatar_yf = yi - distance
                                avatar_xi = xf + distance
                                avatar_xf = xf + distance + avatar_width

                                line_xi = xf
                                line_xf = xf + distance
                                line_yi = yi
                                line_yf = yi - distance

                                draw_avatar = True
                            elif left:
                                avatar_yi = yi - distance - avatar_length
                                avatar_yf = yi - distance
                                avatar_xi = xi - distance - avatar_width
                                avatar_xf = xi - distance

                                line_xi = xi
                                line_xf = xi - distance
                                line_yi = yi
                                line_yf = yi - distance

                                draw_avatar = True
                            else:
                                # Put in the top middle if there's space
                                center_x = xi + (xf - xi) // 2
                                avatar_xi = center_x - avatar_width // 2
                                avatar_xf = center_x + avatar_width // 2
                                avatar_yi = yi - distance - avatar_length
                                avatar_yf = yi - distance

                                if avatar_xi >= 0 and avatar_xf <= frame_width:
                                    line_xi = center_x
                                    line_xf = center_x
                                    line_yi = yi
                                    line_yf = yi - distance
                                    draw_avatar = True
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
                                avatar_xi = xi - distance - avatar_width
                                avatar_xf = xi - distance

                                line_xi = xi
                                line_xf = xi - distance
                                line_yi = yf
                                line_yf = yf + distance

                                draw_avatar = True
                            else:
                                # Put in the bottom middle if there's space
                                center_x = xi + (xf - xi) // 2
                                avatar_xi = center_x - avatar_width // 2
                                avatar_xf = center_x + avatar_width // 2
                                avatar_yi = yf + distance
                                avatar_yf = yf + distance + avatar_length

                                if avatar_xi >= 0 and avatar_xf <= frame_width:
                                    line_xi = center_x
                                    line_xf = center_x
                                    line_yi = yf
                                    line_yf = yf + distance
                                    draw_avatar = True

                        # If we can draw the avatar, do it
                        if draw_avatar:
                            # Make sure we're not out of bounds
                            if (0 <= avatar_yi < frame_length and 0 <= avatar_yf < frame_length and
                                0 <= avatar_xi < frame_width and 0 <= avatar_xf < frame_width):
                                # Place the avatar in the frame
                                try:
                                    frame[avatar_yi:avatar_yf, avatar_xi:avatar_xf] = avatar
                                except ValueError as e:
                                    # Handle dimension mismatch errors
                                    pprint(f"Error placing avatar: {e}")

                                # Draw connecting line
                                cv2.line(frame, (line_xi, line_yi), (line_xf, line_yf), color, thickness)

                                # Draw rectangle around avatar
                                cv2.rectangle(frame, (avatar_xi, avatar_yi), (avatar_xf, avatar_yf), color, thickness)

                                # Display info if available
                                if name in info_social['name'].values:
                                    person_info = info_social[info_social['name'] == name].iloc[0]
                                    text_x = avatar_xf + 10

                                    # Make sure text stays in frame
                                    if text_x + 200 >= frame_width:
                                        text_x = avatar_xi - 210

                                    cv2.putText(frame, name, (text_x, avatar_yi + 20),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)
                                    cv2.putText(frame, f'Birth: {person_info.birthday}',
                                            (text_x, avatar_yi + 45), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                                    cv2.putText(frame, f'FB: {person_info.fb}',
                                            (text_x, avatar_yi + 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                                    cv2.putText(frame, f'IG: {person_info.ig}',
                                            (text_x, avatar_yi + 85), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                                else:
                                    text_x = avatar_xf + 10
                                    if text_x + 100 >= frame_width:
                                        text_x = avatar_xi - 110
                                    cv2.putText(frame, name, (text_x, avatar_yi + 20),
                                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)

                    else:
                        pprint(f"Could not load avatar for {name}")
                else:
                    pprint(f"Avatar not found for {name}")

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    pprint("\n-- Recognition stopped")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pprint(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
