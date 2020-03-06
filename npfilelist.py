#!/usr/bin/env python3

from pathlib import Path
from nputils import natural_sort_key
import logging


class FileList:
    '''
    get all files in same folder as fp,
    make sorted rotatable list,
    shift fp to start
    '''

    def __init__(self, fp, extensions):
        logging.info(f"load filelist {fp}")
        self.current = Path(fp)
        self.extensions = extensions
        self.first = None
        self.last = None
        self.next = None
        self.previous = None
        self._get_files()

    def _get_files(self):

        Fs = self.current.parent.glob("*.*")
        
        fs = [str(f) for f in Fs if f.suffix.lower() in self.extensions]
        
        if len(fs) <= 1:
            return
        
        fs = sorted(fs, key=natural_sort_key)

        self.first = fs[0]
        self.last = fs[-1]

        i = fs.index(str(self.current))
        if i <= len(fs)-2:
            self.next = fs[i+1]
        else:
            self.next = fs[0]
        if i >= 1:
            self.previous = fs[i-1]
        else:
            self.previous = fs[-1]
        self.filelist = fs

#        print(fs)
#        print(i)
#        dfs = deque(fs)
#        print(dfs)
        # shift current to start
#        dfs.rotate(-i)    # rotate to left

#        self.next = dfs[1]
#        self.previous = dfs[-1]
#        print(dfs)
#        return dfs


    def __str__(self):
        from pprint import pformat
        return pformat(self.filelist)
