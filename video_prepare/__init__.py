import os

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)),'')
print(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
print(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
