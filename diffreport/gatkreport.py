import re
from textfile import TextFile, ParseError
from gatktable import GATKTable

class GATKReport:
    def parse(self, fp):
        # #:GATKReport.v1.1:<num tables>
        line = fp.readline()
        fields = re.split(r':', line)

        if len(fields) != 3:
            raise ParseError(fp, "invalid number of fields in report header")

        if fields[0] != "#":
            raise ParseError(fp, "invalid field delimiter in report header")

        if not fields[1].startswith("GATKReport."):
            raise ParseError(fp, "invalid version string in report header")

        self.version = fields[1]
        self.num_tables = int(fields[2])

        for t in xrange(self.num_tables):
            # parse the table
            table = GATKTable(fp)
            # and store it
            self.tables.append(table)

    def __init__(self, filename=None):
        # #:GATKReport.v1.1:<num tables>
        self.version = ""
        self.num_tables = 0
        self.tables = list()

        self.filename = filename
        if filename != None:
            self.parse(TextFile(filename))

    def get_table(self, name):
        for t in self.tables:
            if t.name == name:
                return t

        return None

def load(filename):
    return GATKReport(filename)
