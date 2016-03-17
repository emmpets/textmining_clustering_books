__author__ = 'Manos'

import os
import codecs
import copy
from HTMLParser import HTMLParser
from BeautifulSoup import BeautifulSoup
from re import sub
from sys import stderr
from traceback import print_exc
import shutil

path=r"/Users/Manos/Desktop/gap-html"
# print (os.listdir(path))


# cleaning HTML files
class _DeHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = sub('[ \t\r\n]+', ' ', text)
            self.__text.append(text + ' ')

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.__text.append('\n\n')
        elif tag == 'br':
            self.__text.append('\n')


    def handle_startendtag(self, tag, attrs):
        if tag == 'br':
            self.__text.append('\n\n')



    def text(self):
        return ''.join(self.__text).strip()
def dehtml(text):
    try:
        parser = _DeHTMLParser()
        parser.feed(text)
        parser.close()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


#everything split
for root, dirs, files in os.walk(path):
    # print path to all subdirectories first.
    for file in files:
        print("File = %s" %file)

#only roots
root=next(os.walk(path))[0]
print("Roots = %s"%root)

#only dirs
dirs=next(os.walk(path))[1]
print("Dirs = %s"%dirs)


#only files
files=next(os.walk(path))[2]
if '.DS_Store' in files:
    files.remove('.DS_Store')
print("Files = %s"%files)




for dirs in os.listdir(path):
    if not os.path.isdir(path+'/'+dirs):
        continue

    folders = copy.deepcopy(dirs)
    print(folders)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for files in os.listdir(path+'/'+folders):

        f=codecs.open(path+'/'+folders+'/'+files, 'r')

        with open(folders+ ".txt", 'a') as outfile:
            contents = ""
            if f is not None:
                contents=f.read()
                contents_after_OCR=dehtml(contents).replace ("OCR Output", "")
                # print contents
                # print(contents_after_OCR)
                contents = contents + " " +contents_after_OCR
                with open(folders+ ".txt", "a") as f:
                    f.write(contents_after_OCR)

