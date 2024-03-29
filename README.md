# Face Recognition

Build your personal face detection and face recognition algorithm. 

<br>
<p align="center">
  <img width="640"  src="./images/test_01.png">
</p>  
<br>
<br>
<p align="center">
  <img width="400"  src="./images/test_02.png">
  <img width="400"  src="./images/test_03.png">
  <img width="400"  src="./images/test_04.png">
</p>  
<br>

## Python Version

This code i tested to work with Pytohn 3.9.

## Environment

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to

- Database creation

In order to prepare your personal database put your images (in jpg format) inside the `known_people` folder. Each picture must contain no more than one face and the file name has to be reflect the person represented. File name will be use as label.

- Run

```
python app.py
```

- Optional arguments
```
usage: app.py [-h] [-c CAMERA] [-k KNOWN] [-r SCALEDOWN]

optional arguments:
  -h, --help            show this help message and exit
  -c CAMERA, --camera_id CAMERA
                        ID of the camera. An integer between 0 and N. Default
                        is 1
  -k KNOWN, --known KNOWN
                        folder with the known people images
  -r SCALEDOWN, --scale SCALEDOWN
                        scaling factor: 0.25 means 25 percent of the original
                        size.
```

## Bibliography

- https://www.youtube.com/watch?v=535acCxjHCI

- https://www.youtube.com/watch?v=d2QIw6cQg40

- https://www.youtube.com/watch?v=5yPeKQzCPdI



