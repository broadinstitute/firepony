# wrapper for reading text files line by line, tracks the filename and current line number
class TextFile:
    def __init__(self, filename):
        self.__filename = filename
        self.__fp = open(self.__filename)
        self.__line = 0

    def readline(self):
        ret = self.__fp.readline().rstrip()
        self.__line += + 1
        return ret

    def lineno(self):
        return self.__line

    def filename(self):
        return self.__filename

# parse error exception, contains the line number
class ParseError(Exception):
    def __init__(self, fp, error=None):
        self.__filename = fp.filename()
        self.__lineno = fp.lineno()
        self.__error = error

    def __str__(self):
        ret = "Parse error on " + self.__filename + ":" + self.__lineno
        if self.__error != None:
            ret = ret + ": " + self.__error

        return ret.encode("ascii", "backslashreplace")

    def __repr__(self):
        return self.__str__()
