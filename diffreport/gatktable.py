import re
import operator
from textfile import ParseError

class GATKTable:
    def __read_data_line(self, fp):
        line = fp.readline()
        # remove whitespace at beginning of line
        line = re.sub(r'^\s+', '', line)
        # transform whitespace into a single field separator
        line = re.sub(r'\s+', '\x00', line)
        fields = re.split(r'\x00', line)

        if len(fields) != self.num_columns:
            raise ParseError(fp, "invalid number of columns")

        return fields

    def __parse_table_header(self, fp):
        # parse table descriptor
        # #:GATKTable:<number of fields>:<number of rows>:<format specifier>:...:<format specifier>:;
        line = fp.readline()
        fields = re.split(r':', line)

        if len(fields) < 6:
            raise ParseError(fp, "not enough fields in table header")

        if fields[0] != "#":
            raise ParseError(fp, "first field in table header is invalid")

        if fields[1] != "GATKTable":
            raise ParseError(fp, "second field in table header is invalid")

        if fields[-1] != ";":
            raise ParseError(fp, "table header terminator not found")

        try:
            self.num_columns = int(fields[2])
        except:
            raise ParseError(fp, "invalid column count in table header")

        try:
            self.num_rows = int(fields[3])
        except:
            raise ParseError(fp, "invalid row count in column header")

        # now parse all the format strings into tuples of the form (<digit specifier>, <format character>)
        for field in fields[4:-1]:
            m = re.match(r'^\%(\.([0-9]+))?([sfd])$', field)
            if m == None:
                raise ParseError(fp, "error parsing field format specifier '" + field + "'")

            fmt = m.groups()[1:]
            self.field_format_specifiers.append(fmt)

        # parse table name and description
        #:GATKTable:Arguments:Recalibration argument collection values used in this run
        line = fp.readline()
        fields = re.split(r':', line)

        if len(fields) != 4:
            raise ParseError(fp, "not enough fields in table description")

        if fields[0] != "#":
            raise ParseError(fp, "first field in table description is invalid")

        if fields[1] != "GATKTable":
            raise ParseError(fp, "second field in table description is invalid")

        self.name = fields[2]
        self.description = fields[3]

        # finally parse the first row of the table, containing the column names
        self.column_names = self.__read_data_line(fp)

    def parse(self, fp):
        # read table header
        self.__parse_table_header(fp)

        for r in xrange(self.num_rows):
            # parse each entry in the table according to the format specifier
            fields = self.__read_data_line(fp)
            record = list()

            for i in xrange(len(self.field_format_specifiers)):
                desc = self.field_format_specifiers[i][1]
                data = fields[i]

                if desc == "s":
                    # read field as string
                    record.append(data)
                elif desc == "d":
                    # read field as integer
                    record.append(int(data))
                elif desc == "f":
                    # read field as floating point
                    record.append(float(data))
                else:
                    raise ParseError(fp, "unknown format field specifier " + desc)

            self.rows.append(record)

        # consume the empty line at the end
        line = fp.readline()
        if line != "":
            raise ParseError(fp, "table delimiter not found")

    # sorts the table using all string and integer fields as keys
    def __sort(self):
        # build list of keys
        keys = list()

        for i in xrange(len(self.field_format_specifiers)):
            desc = self.field_format_specifiers[i][1]
            if desc == "s" or desc == "d":
                keys.append(i)

        self.rows = sorted(self.rows, key = operator.itemgetter(*keys))

    def __init__(self, fp=None):
        # #:GATKTable:2:17:%s:%s:;
        self.num_columns = 0
        self.num_rows = 0
        self.field_format_specifiers = list()

        # #:GATKTable:Arguments:Recalibration argument collection values used in this run
        self.name = ""
        self.description = ""

        # Argument      Value
        self.column_names = list()

        # list of rows, each row is a list of columns
        self.rows = list()

        if fp != None:
            self.parse(fp)
            self.__sort()
