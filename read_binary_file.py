import numpy as np
import struct

'''
# read binary file from MODFLOW based on
The array data can be either in single-precision or double-precision format.
There is no data field that indicates which format is used.  Instead, you will need to try both and see which one works.
First read the following variables in order
KSTP: the time step number, an integer, 4 bytes.
KPER: the stress period number, an integer, 4 bytes.
PERTIM: the time in the current stress period, a real number, either 4 or 8 bytes.
TOTIM, the total elapsed time, a real number, either 4 or 8 bytes.
DESC, a description of the array, 16 ANSI characters, 16 bytes.
NCOL, the number of columns in the array, an integer, 4 bytes.
NROW, the number of rows in the array, an integer, 4 bytes.
ILAY, the layer number, an integer, 4 bytes.
Next come a list of NROW x NCOL real numbers that represent the values of the array.
The values are in row major order.  Each value in the array occupies either 4 or 8 bytes
depending on whether the values are in single- or double-precision.
After reading one set of values, start over with KSTP. Continue until reaching the end of the file.
The following is a list of possible values for DESC.  The list may be incomplete and is subject to change.
Check the values passed to the subroutine ULASAV in the MODFLOW source code for other possible values.
'            HEAD'
'        DRAWDOWN'
'      SUBSIDENCE'
'      COMPACTION'
'   CRITICAL HEAD'
'     HEAD IN HGU'
'NDSYS COMPACTION'
'  Z DISPLACEMENT'
' D CRITICAL HEAD'
'LAYER COMPACTION'
' DSYS COMPACTION'
'ND CRITICAL HEAD'
'LAYER COMPACTION'
'SYSTM COMPACTION'
'PRECONSOL STRESS'
'CHANGE IN PCSTRS'
'EFFECTIVE STRESS'
'CHANGE IN EFF-ST'
'      VOID RATIO'
'       THICKNESS'
'CENTER ELEVATION'
'GEOSTATIC STRESS'
'CHANGE IN G-STRS'
One way to determine whether the file has been saved with single- or double-precision,
is to read the file up through DESC using either single- or  double-precision numbers and
see if the value read for DESC matches one of the above values.
'''


def read_binary_file(f, nrow, ncol, nlay,ntrans):
    ffile = open(f, 'rb')
    # result = np.zeros([nsp * 5, nrow, ncol])  # 3 for subsidence
    head = np.zeros((nrow, ncol, nlay * ntrans))
    for i in range(nlay*ntrans):  # 3 for subsidence
        # print 'i',i
        KSTP = struct.unpack('i', ffile.read(4))[0]
        #print('KSTP:',KSTP)
        KPER = struct.unpack('i', ffile.read(4))[0]
        #print('KPER:',KPER)
        PERTIM = struct.unpack('f', ffile.read(4))[0]
        #print('PERTIM:',PERTIM)
        TOTIM = struct.unpack('f', ffile.read(4))[0]
        #print('TOTIM:',TOTIM)
        DESC = ''
        for ii in range(16):
             DESC+=  ffile.read(1).decode(encoding='UTF-8')
        #print('DESC:',DESC)
        #ffile.seek(16, 1)
        NCOL = struct.unpack('i', ffile.read(4))[0]
        # print('NCOL:',NCOL)
        NROW = struct.unpack('i', ffile.read(4))[0]
        # print('NROW:',NROW)
        NLAY = struct.unpack('i', ffile.read(4))[0]
        #print('NLAY:',NLAY)
        # ffile.seek(12,1)
        ncell = NCOL * NROW
        temp = np.zeros(ncell)
        for j in range(ncell):
             temp[j] = struct.unpack('f', ffile.read(4))[0]
        result_plan = temp.reshape(NROW, NCOL)
        #print('i ',i)
        head[:, :, i] = result_plan
        # if i<nrow*ncol-1:
        #    for i in range(12):
        #        t=struct.unpack('s',ffile.read(1))[0]
    length = ffile.tell()
    print('length:',length)
    ffile.close()
    '''
    for i in range(nrow):
        for j in range(ncol):
            if result[nsp*3,]
    '''
    max_head = np.max(head)
    i, j, k = np.unravel_index(head.argmax(), head.shape)  # in order to find the place of the maximum subsidence
    print('max_head:',max_head)
    # print i
    # print j
    # print k
    return head

