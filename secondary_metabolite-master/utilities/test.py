import os
import subprocess

from dataclasses import dataclass

@dataclass
class Test(object):

    filename: str
    path: str 

    def __post_init__(self):
        sec_path: str = os.path.join(os.getcwd(), '../tmp/minimal_genome/{}.hmm'.format(self.filename))
        subprocess.Popen(['hmmbuild', sec_path, self.path], stdout=subprocess.PIPE)

path: str = os.path.join(os.getcwd(), '../tmp/tmp')

for(dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        tmp_path: str = os.path.join(path, filename)
        if os.path.isfile(tmp_path):
            test = Test(filename, tmp_path)