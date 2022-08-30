import instaloader

L = instaloader.Instaloader()

username = input('Enter your Instagram username: ')
password = input('Enter your Instagram password: ')

# Login or load session
L.login(username, password)        # (login)
print('\nConnection successful!')

# Obtain profile metadata
profile = instaloader.Profile.from_username(L.context, username)
# Print list of followees

followers = [i.username for i in profile.get_followers()]

for i in followers:
    print(i)
    try:
        instaloader.Instaloader().download_profile(i, profile_pic_only=True)
    except Exception:
        pass
