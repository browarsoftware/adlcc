import os
from fnmatch import fnmatch

root = 'd:\\dane\\CASIA-WebFace\\'
pattern = "*.jpg"

file_name = 'identity_CASIA-WebFace.txt'

file_object = open(file_name,'w')
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            #print(name)
            x = path.split("\\")
            identity = x[len(x) - 1]
            #print(identity)
            #print(os.path.join(path, name))
            str_to_save = str(identity) + "\\" + str(name) + " " + str(identity)
            #print(str_to_save)
            file_object.write(str_to_save)
            file_object.write('\n')
file_object.close()