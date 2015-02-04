#!/usr/bin/env python

import sys
import math
import collections
from StringIO import StringIO

import gatkreport

# define an error type
ErrorBase = collections.namedtuple('Error', [
        'table_name',       # name of the table where error was found
        'column_name',      # name of the column with error
        'table_row',        # offending row
        'table_col',        # offending column
        'val_A',            # left value
        'val_B',            # right value
        'delta',            # absolute difference between both values
        'relative_delta',   # relative difference between both values
        'data_A',           # full row of data, left table
        'data_B',           # full row of data, right table
        'table_A',          # left table
        'table_B',          # right table
])

ErrorBase.__new__.__defaults__ = (
    None,                   # table_name
    None,                   # column_name
    None,                   # table_row
    None,                   # table_column
    None,                   # val_A
    None,                   # val_B
    0.0,                    # delta
    0.0,                    # relative_delta
    None,                   # data_A
    None,                   # data_B
    None,                   # table_A
    None,                   # table_B
)

class Error (ErrorBase):
    __slots__ = ()

    def __str__(self):
        s = StringIO()

        print >> s, "%s: error = %f%% (row %d col %d left=%f right=%f abs delta=%f)" % (
            err.table_name,
            err.relative_delta,
            err.table_row,
            err.table_col,
            err.val_A,
            err.val_B,
            err.delta
        )

        print >> s, " left: ",
        for field in err.data_A:
            print >> s, str(field) + " ",

        print >> s, ""

        print >> s, " right: ",
        for field in err.data_B:
            print >> s, str(field) + " ",

        return s.getvalue()

    # logic to decide whether a given error can be waived
    def can_waive(self):
        if self.delta == 0.0:
            return True

        if self.relative_delta >= 0.5:
            if self.table_A.column_names[self.table_col] == "EmpiricalQuality":
                # empirical quality is a discretized value, which amplifies small errors in values around the .5 threshold
                # waive off-by-one errors in empirical quality
                if math.fabs(math.fabs(self.val_B) - math.fabs(self.val_A)) > 1.0:
                    error_index = self.table_A.column_names.index("Error")
                    error_A = self.data_A[error_index]
                    error_B = self.data_B[error_index]

                    frac_A = math.fabs(math.modf(error_A)[0])
                    frac_B = math.fabs(math.modf(error_B)[0])

                    dist_A = 0.5 - frac_A
                    dist_B = 0.5 - frac_B

                    if (dist_A < 0.0 and dist_B >= 0.0) or (dist_A >= 0.0 and dist_B < 0.0):
                        if math.fabs(dist_A) < 0.01 and math.fabs(dist_B) < 0.01:
                            return True
        else:
            # less than 0.5% is waivable
            return True

        # everything else is a real error
        return False

    @staticmethod
    def diff(table_A, table_B, row, col):
        val_A = table_A.rows[row][col]
        val_B = table_B.rows[row][col]
        delta = max(math.fabs(val_A), math.fabs(val_B)) - min(math.fabs(val_A), math.fabs(val_B))
        if delta != 0.0 and max(math.fabs(val_A), math.fabs(val_B)) != 0.0:
            relative_delta = (delta / max(math.fabs(val_A), math.fabs(val_B))) * 100.0
        else:
            relative_delta = 0.0

        err = Error(table_name = table_A.name,
                    column_name = table_A.column_names[col],
                    table_row = row,
                    table_col = col,
                    val_A = val_A,
                    val_B = val_B,
                    delta = delta,
                    relative_delta = relative_delta,
                    data_A = table_A.rows[row],
                    data_B = table_B.rows[row],
                    table_A = table_A,
                    table_B = table_B)

        return err

if len(sys.argv) != 3:
    print "Usage: %s <report-1> <report-2>" % (sys.argv[0])
    sys.exit(1)

fname_A = sys.argv[1]
fname_B = sys.argv[2]

report_A = gatkreport.load(fname_A)
report_B = gatkreport.load(fname_B)

exit_code = 0

# iterate through the list of tables in A
for table_A in report_A.tables:
    # find corresponding table in B
    table_B = report_B.get_table(table_A.name)
    if table_B == None:
        continue

    if table_A.field_format_specifiers != table_B.field_format_specifiers:
        print "field format specifier mismatch"
        sys.exit(1)

    # compute the set of directly comparable fields vs fields that require an error metric
    compare_indices = list()
    measure_indices = list()
    for i in xrange(len(table_A.field_format_specifiers)):
        # floats are not comparable
        if table_A.field_format_specifiers[i][1] == "f":
            measure_indices.append(i)
        else:
            compare_indices.append(i)

    # now diff the list based on the indices
    max_error = Error(delta=0.0)

    for i in xrange(table_A.num_rows):
        # compare everything that should match exactly
        for j in compare_indices:
            if table_A.rows[i][j] != table_B.rows[i][j]:
                print "Mismatch on file %s table %s row %d column %d" % (fname_B, table_A.name, i, j)
                sys.exit(1)

        # measure the error for floating point values
        for j in measure_indices:
            err = Error.diff(table_A, table_B, i, j)

            if err.delta > 0.0:
                if not err.can_waive():
                    #print "*** ERROR"
                    print err
                    exit_code = 2

if exit_code == 0:
    print "no errors"

sys.exit(exit_code)
