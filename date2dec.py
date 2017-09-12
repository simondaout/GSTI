from datetime import datetime
import numpy as np

def date2dec(dates):
    dates  = np.atleast_1d(dates)
    times = []
    for date in dates:
        x = datetime.strptime('{}'.format(date),'%Y%m%d')
        dec = float(x.strftime('%j'))/365.1
        year = float(x.strftime('%Y'))
        # print date,dec,year
        times.append(year + dec)
    return times

def time2dec(dates):
    dates  = np.atleast_1d(dates)
    times = []
    for date in dates:
        x = datetime.strptime('{}'.format(date),'%Y-%m-%d %H:%M:%S.%f')
        year = float(x.strftime('%Y'))
        day = float(x.strftime('%j'))
        hour = float(x.strftime('%H'))
        minutes = float(x.strftime('%M'))
        sec = float(x.strftime('%S.%f'))

        dec = (day*24*60*60 + hour*60*60 + minutes*60 + sec) / (365.1*24*60*60)
        # print date,dec,year
        times.append(year + dec)
    return times