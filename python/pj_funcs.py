# Import[ant] libaries @('_')@
import hl7
import pandas as pd
import numpy as np
import re
import os
import math
import time
import datetime

# Plotting!
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors





###################################################

def NoError(func, *args, **kw):
    '''
    Determine whether or not a function and its arguments gives an error
    For purposes of this HL7 Project, it is typically used in conjunction with the functions index(),index_n(), or exec()
    
    Parameters
    ----------
    func: function, required
    *args: varies, required
    
    Returns
    -------
    bool
        True if function does not cause error.
	False if function causes error.
        
    Requirements
    ------------
    -none
    '''
    try:
        func(*args, **kw)
        return True
    except Exception:
        return False
    
def index(m,ind):
    '''
    Simple function to return m[ind]
    For purposes of this HL7 parsing project, this is typically used in conjunction with the NoError() function.
    '''
    return m[ind]

def LIKE(array,word):
    '''
    Finds all parts of list that have a word in them
    
    Parameters
    ----------
    array : list/array type, required
    word : str, required
    
    Returns
    -------
    np.array
        An array which is a subset of the original containing the word
        
    Requirements
    ------------
    -import numpy as np
    
    '''
    # Convert to numpy array.  Everything's easier with numpy
    array = np.array(array)
    
    # Create in-condition.  List of True/False for each element
    cond = np.array([str(word) in array[i] for i in np.arange(0,len(array))])
    
    # Enact that condition 
    subset = array[cond]
    
    # Return the subset
    return subset

###################################################

def completeness_facvisits(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe outputted from NSSP_Element_Grabber() function.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits
    3. Return Dataframe.
        dataframe.index -> Facility Name, Number of Visits
        dataframe.frame -> Percents of visits within hospital with
            non-null values in specified column
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        should have format outputted from NSSP_Element_Grabber() function
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -from pj_funcs import *
 
    '''

    start_time = time.time()
    
    # Make a visit indicator that combines facility|mrn|visit_num
    df['VISIT_INDICATOR'] = df[['FACILITY_NAME', 'PATIENT_MRN', 'PATIENT_VISIT_NUMBER']].astype(str).agg('|'.join, axis=1)

    # Create array of Falses.  Useful down the road 
    false_array = np.array([False] * len(df.columns))

    # Create empty dataframe we will eventually insert into
    empty = pd.DataFrame(columns=df.columns)

    # Create empty lists for facility_names (facs) and number of patients in a facility (num_patients)
    # These lists will serve as our output's descriptive indexes
    num_visits = []
    facs = []

    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = df.groupby('FACILITY_NAME',sort=False)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

        # Append facility name to empty list
        facs.append(facility)

        # Initiate visit count
        visit_count = 0

        # Sort by Patient MRN
        MRN_sort = df1.groupby(['VISIT_INDICATOR'],sort=False)

        # Initiate list of 0s.  Each column gets +1 for each visit with a non-null column value.
        countz = false_array.copy().astype(int)

        for visit, df3 in MRN_sort:


            # Initiate array of falses
            init = false_array.copy()

            # Looping through the visits ADT data rows, look for non_null values.  True if non-null. 
            #       Use OR-logic to replace 0s in init with 1s and keep 1s as 1s for each iterated row.
            for i in np.arange(0,len(df3)):
                init = init | (df3.iloc[i].notnull())

            # Add information on null (0) vs. non-null (1) columns to countz which is initially all 0 but updates for each patient.
            countz += init.astype(int)

            # Show that the number of visits has increased
            visit_count += 1


        # Append visit number to empty list
        num_visits.append(visit_count)

        # Update empty dataframe with information on completeness (out of 100%) we had for each column
        # * note countz is a 1D array that counts how many visits have non-null values in each column.
        empty.loc[facility,:] = (countz/visit_count)*100


    # Clarify and Create index information for output Dataframe
    empty['Num_Visits'] = num_visits
    empty['Facility'] = facs
    empty = empty.set_index(['Facility','Num_Visits'])
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty

################################################################


def tattle(hosp,issue,num,frac):
    '''
    Function that writes to a "Tattle" log.
    If you ever find improper formatting/issues that screw up your code, run this
         on your scripts and want to tattle about it.  Will send to a tattle log.

    Inputs
    -------
    - facility
    - brief issue description (try to be uniform)
    - number of issues
    - fraction of total cases that have issues
    
    Outputs
    -------
    None (simply writes to file)
    '''
    # Clean up
    frac = round(frac,3)
    num = int(num)
    issue = str(issue)
    hosp = str(hosp)
    
    cols = ['Facility','Issue','Num_Issues','Total_issue_percent']
    newrow = pd.DataFrame(columns=cols)
    newrow.loc[0] = [hosp,issue,num,frac]
    newrow.to_csv('../data/processed/tattle_sheet.csv',mode='a',header=False, index=False)
    

################################################################

def to_hours(item):
    '''
    Takes a datetime object and converts them to the time in hours,
    as a float rounded to the 3rd decimal.
    
    Input
    -----
    item - DateTime object, required
    
    Output
    -----
    Time in hours (dtype: Float)
    
    Requirements
    ------------
    *Libraries*
    -import datetime
    
    *Functions*
    none
    
    '''
    return round((datetime.timedelta.total_seconds(item) / (60*60)),3)

#####################################################################

def to_days(item):
    '''
    Takes a datetime object and converts them to the time in days,
    as a float rounded to the 3rd decimal.
    
    Input
    -----
    item - DateTime object, required
    
    Output
    -----
    Time in days (dtype: Float)
    
    Requirements
    ------------
    *Libraries*
    -import datetime
    
    *Functions*
    none
    
    '''
    return round((datetime.timedelta.total_seconds(item) / (24*60*60)),3)

####################################################################

def timeliness_facvisits_days(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe straight from PHESS SQL Query-pulled file.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits.  
    3. Return Dataframe
        dataframe.index -> Facility Name
        dataframe.frame -> Statistics on time differences between MSG_DATETIME
                            and ADMIT_DATETIME
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        example:  df = pd.read_csv('some/path/PHESS_OUTPUT_FILE.csv', encoding = 'Cp1252')
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -import pandas as pd
    -import numpy as np
    -import datetime
    -import time
    -from pj_funcs import *    

    *Functions*
    - to_days    (found in pj_funcs.py file)

    '''

    start_time = time.time()

    # Cleanup 1:  ADMIT_DATETIME == 'Missing admit datetime'
    df = df[df['ADMIT_DATETIME'] != 'Missing admit datetime']

    # Cleanup 2:  Some datetimes (meaning 1/1000+) have a decimal in them
    #           They cannot be interpreted as datetimes via pd.to_datetime
    #           so we need to convert them.

    # Interperet ADMIT_DATETIME as string
    admit_time = df['ADMIT_DATETIME'].astype(str)

    # Use Pandas str.split function to divide on decimal, expand, and
    #      take the first argument (everything before the decimal).
    admit_time = admit_time.str.split('\.',expand=True)[0]

    # Convert our newly cleaned strings to datetime type. For uniformity, choose UTC
    admit_time = pd.to_datetime(admit_time, utc=True)

    # Do the exact same thing to 'MSG_DATETIME'
    msg_time = df['MSG_DATETIME'].astype(str)
    msg_time = msg_time.str.split('\.',expand=True)[0]
    msg_time = pd.to_datetime(msg_time, utc=True)

    # Update 'ADMIT_DATETIME' and 'MSG_DATETIME' columns to new format
    df['ADMIT_DATETIME'] = admit_time
    df['MSG_DATETIME'] = msg_time
    
    ##################################################################
    
    #  Create TimeDif Column!!

    TimeDif = msg_time - admit_time

    #  Apply my personal to_days function to see datetime differences in days.
    #  Information can be found in pj_funcs.py or by typing 'to_days?' in a cell
    df['TimeDif (days)'] = TimeDif.apply(to_days)
    

    # Only take the important columns in sub-dataframe
    sub_df = df[['ADMIT_DATETIME','MSG_DATETIME','PATIENT_MRN',
                           'PATIENT_VISIT_NUMBER','FACILITY_NAME','TimeDif (days)']]


    ##################################################################
    
    facs = []


    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = sub_df.groupby('FACILITY_NAME',sort=False)

    # Label columns we will eventully populate in empty dataframe
    stats_cols = ['Num_Visits','Median','Avg','StdDev','Min','Max']
    empty = pd.DataFrame(columns=stats_cols)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

            # Create empty list to fill with TimeDif (days) values for visits
            fillme = []

            # Sort by Patient MRN
            MRN_sort = df1.groupby(['PATIENT_MRN'],sort=False)

            # Loop through MRN groupings
            for patient, df2 in MRN_sort:

                # If there is a null value in the MRN group, we have a problem
                if sum(df2['PATIENT_VISIT_NUMBER'].isnull()) > 0:

                    # If there is only one row and its null, its one patient.
                    if len(df2) == 1:
                        fillme.append(df2.iloc[0]['TimeDif (days)'])

                # Cases where all PATIENT_VISIT_NUMBER are non-null!
                else:

                    # Sort further by Patient Visit Number
                    VisNum_sort = df2.groupby(['PATIENT_VISIT_NUMBER'],sort=False)

                    # Loop through Patient Visit Numbers
                    for visit, df3 in VisNum_sort:

                        # Find the row with the newest 
                        index_earliest = df3['ADMIT_DATETIME'].idxmin()

                        # Within our early admit datetime row, pull TimeDif
                        dif_we_take = df3.loc[index_earliest]['TimeDif (days)']

                        # Append correct TimeDif to fillme list
                        fillme.append(dif_we_take)

            # Convert list (that we appended to) into np array and perform stats
            fillme = np.array(fillme)

            stats = [len(fillme),np.median(fillme),np.mean(fillme),np.std(fillme),
                    np.min(fillme),np.max(fillme)]

            # Fill stats into dataframe for that facility.  Rounded to 2 decimals
            empty.loc[facility,:] = np.array(stats).round(2)
        
        
    ###########################################################################
    
    
    
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty



##############################################################################################################################


def timeliness_facvisits_hours(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe straight from PHESS SQL Query-pulled file.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits.  
    3. Return Dataframe
        dataframe.index -> Facility Name
        dataframe.frame -> Statistics on time differences between MSG_DATETIME
                            and ADMIT_DATETIME
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        example:  df = pd.read_csv('some/path/PHESS_OUTPUT_FILE.csv', encoding = 'Cp1252')
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -import pandas as pd
    -import numpy as np
    -import datetime
    -import time
    
    *Functions*
    - to_hours    (found in pj_funcs.py file)

    '''

    start_time = time.time()

    # Cleanup 1:  ADMIT_DATETIME == 'Missing admit datetime'
    df = df[df['ADMIT_DATETIME'] != 'Missing admit datetime']

    # Cleanup 2:  Some datetimes (meaning 1/1000+) have a decimal in them
    #           They cannot be interpreted as datetimes via pd.to_datetime
    #           so we need to convert them.

    # Interperet ADMIT_DATETIME as string
    admit_time = df['ADMIT_DATETIME'].astype(str)

    # Use Pandas str.split function to divide on decimal, expand, and
    #      take the first argument (everything before the decimal).
    admit_time = admit_time.str.split('\.',expand=True)[0]

    # Convert our newly cleaned strings to datetime type. For uniformity, choose UTC
    admit_time = pd.to_datetime(admit_time, utc=True)

    # Do the exact same thing to 'MSG_DATETIME'
    msg_time = df['MSG_DATETIME'].astype(str)
    msg_time = msg_time.str.split('\.',expand=True)[0]
    msg_time = pd.to_datetime(msg_time, utc=True)

    # Update 'ADMIT_DATETIME' and 'MSG_DATETIME' columns to new format
    df['ADMIT_DATETIME'] = admit_time
    df['MSG_DATETIME'] = msg_time
    
    ##################################################################
    
    #  Create TimeDif Column!!

    TimeDif = msg_time - admit_time

    #  Apply my personal to_days function to see datetime differences in days.
    #  Information can be found in pj_funcs.py or by typing 'to_days?' in a cell
    df['TimeDif (hrs)'] = TimeDif.apply(to_hours)
    

    # Only take the important columns in sub-dataframe
    sub_df = df[['ADMIT_DATETIME','MSG_DATETIME','PATIENT_MRN',
                           'PATIENT_VISIT_NUMBER','FACILITY_NAME','TimeDif (hrs)']]


    ##################################################################
    
    facs = []


    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = sub_df.groupby('FACILITY_NAME',sort=False)

    # Label columns we will eventully populate in empty dataframe
    stats_cols = ['Num_Visits','Avg TimeDif (hrs)','% visits recieved within 24 hours','% visits recieved between 24 and 48 hours ',
                  '% visits recieved after 48 hours']
    empty = pd.DataFrame(columns=stats_cols)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

            # Create empty list to fill with TimeDif (hrs) values for visits
            fillme = []

            # Sort by Patient MRN
            MRN_sort = df1.groupby(['PATIENT_MRN'],sort=False)

            # Loop through MRN groupings
            for patient, df2 in MRN_sort:

                # If there is a null value in the MRN group, we have a problem
                if sum(df2['PATIENT_VISIT_NUMBER'].isnull()) > 0:

                    # If there is only one row and its null, its one patient.
                    if len(df2) == 1:
                        fillme.append(df2.iloc[0]['TimeDif (hrs)'])

                # Cases where all PATIENT_VISIT_NUMBER are non-null!
                else:

                    # Sort further by Patient Visit Number
                    VisNum_sort = df2.groupby(['PATIENT_VISIT_NUMBER'],sort=False)

                    # Loop through Patient Visit Numbers
                    for visit, df3 in VisNum_sort:

                        # Find the row with the newest 
                        index_earliest = df3['ADMIT_DATETIME'].idxmin()

                        # Within our early admit datetime row, pull TimeDif
                        dif_we_take = df3.loc[index_earliest]['TimeDif (hrs)']

                        # Append correct TimeDif to fillme list
                        fillme.append(dif_we_take)

            # Convert list (that we appended to) into np array and perform stats
            fillme = np.array(fillme)
            
            cond_bottom = (fillme <= 24)
            cond_middle = (fillme > 24)&(fillme < 48)
            cond_top = (fillme >= 48)
            
            percent_bottom = round((sum(cond_bottom)/len(fillme)),3)*100
            percent_middle = round((sum(cond_middle)/len(fillme)),3)*100
            percent_top = round((sum(cond_top)/len(fillme)),3)*100

            stats = [len(fillme),np.mean(fillme),percent_bottom,percent_middle,percent_top]

            # Fill stats into dataframe for that facility.  Rounded to 2 decimals
            empty.loc[facility,:] = np.array(stats).round(2)
        
        
    ###########################################################################
    
    
    
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty


##################################################################################

def index_n(m,ind):
    '''
    Indexes some object 'm' by each element in the list 'ind'
    
    Parameters
    ----------
    m: type varies, required
    ind: list, required
    
    Returns
    -------
    m[ind[0]][ind[1]][ind[...]][ind[n]]
     
    Requirements
    ------------
    -Numpy as np
    
    '''
    for i in np.arange(0,len(ind)):
        m = m[ind[i]]
    return m

###################################################################################

def Index_pull(ind,m):
    
    '''
    Locates and returns the element within a message 'm' thats location
        is described by indeces, 'ind'
    
    Parameters
    ----------
    ind: list, required, full index path as list indicating HL7 location.
    m: hl7 type object, required, m = hl7.parse(some_message)
    
    Returns
    -------
    Str
        Element
     
    Requirements
    ------------
    -NoError from pj_funcs.py
    -index_n from pj_funcs.py
    -hl7
    
    '''
    
    output = ''
    
    # Try indexing the message by ind
    if NoError(index_n,m,ind):
        
        #  If the indexing up to the 2nd to last element returns a string, accept it.  Call it 'output'
        if type(index_n(m,ind[:-1])) == str:
            output = index_n(m,ind[:-1])

        # Normally, we will take the exact, full-indexed value.  Call it 'output'
        else:
            output = str(index_n(m,ind))
    
    # Return output.  If none found, return empty string, ''
    return output

######################################################################################


def Index_pull_CONC(field,rest_index,m):
    '''
    Returns a concetated string for elements with repeating fields. Seperated by '|' characters.
    
    Example: consider the case of Ethnicity Code where a patient may have multiple selected ethnicities.
        For our example we will assume this element is always located in PID-22.1.
    
            print(Index_pull_CONC('PID', [22,0,0], m))
                Ethnicity1|Ethnicity2
        
        Note:  Ethnicity1 and Ethnicity2 are pulled from PID|x|-22.1 and PID|y|-22.1 respectively where
            x,y are non-equal integers representing different repetitions of a repeated field.
        
    
    Parameters
    ----------
    field: list (with one element), required, for non-empty return choose valid 3 letter HL7 field
    rest: list, required, integer list indicating where to find it.
    m: hl7 type object, required, m = hl7.parse(some_message)
    
    Returns
    -------
    Str
        Concetation represented by '|'
     
    Requirements
    ------------
    -NoError from pj_funcs.py
    -index_n from pj_funcs.py
    -Numpy as np
    -hl7
    
    '''
    
    # Initialize empty output
    output = ''
    
    # Read in field
    field_str = field[0]
    
    # Check to see if the field exists in our message
    if NoError(index,m,field_str):
        
        # Set the field equal to 'fi'
        fi = m[field_str]
        
        # If the field repeats, it has a non-zero length. Loop through its length 1 by 1
        for u in np.arange(0,len(fi)):
            
            # Identify the total index by summing strings: field, loop_number, rest_index
            tot_index = field+[u]+rest_index
            
            # Make sure message can be indexed by the total index
            if NoError(index_n,m,tot_index):
                
                #  If the indexing up to the 2nd to last element returns a string, accept it.  Call it 'output'
                if type(index_n(m,tot_index[:-1])) == str:
                    full = index_n(m,tot_index[:-1])
                    
                    # If this string, 'full', has non-zero length, add it to our output and end with '|'
                    if len(full)>0:
                        output += full
                        output += '|'
                        
                # Normally, we will take the exact, full-indexed value.  Call it 'output'
                else:
                    full = str(index_n(m,tot_index))
                    
                    # If this string, 'full', has non-zero length, add it to our output and end with '|'
                    if len(full)>0:
                        output += full
                        output += '|'
                        
                # Go back and loop through more repeated fields until no more exist
                
    # if non-zero length output, clean up last trailing '|' character
    if len(output)>0:
        if output[-1] == '|':
            output = output[:-1]
            
    # Return output.  If none found, this will be '' (empty string)
    return output

############################################################################################################

def DI_One(ind,m,df,z,col_name):
    
    '''
    Returns the element value of 'm' indexed by 'ind'.
    Updates the dataframe 'df' cell value indexed by 'z' and 'col_name'
    
    Parameters
    ----------
    ind: list, required, complete index path (as list) to desired element
    m: hl7 type object, required, m = hl7.parse(some_message)
    df:  pandas DataFrame, required
    z:  int, required, valid integer row index of df
    col_name: str, required, valid column in df
    
    Returns
    -------
    Str
        Element
        
    Output
    ------
    Updates dataframe
        df.loc[z,col_name] = Element
     
    Requirements
    ------------
    -Index_pull from pj_funcs.py
    -Pandas
    -hl7
    
    '''
    
    # Call the index on the message.
    obj = Index_pull(ind,m)
    
    # See if the 'obj' is an actual non-zero thing.
    if len(obj)>0:
        
        # If so, append to the row_z, col_colname in Dataframe, df
        df.loc[z,col_name] = obj
        
    # Else:  Do nothing.
    
    # Return the object.  If none found, will return empty str, '' with no df update
    return obj

####################################################################

def DI_One_CONC(field,ind,m,df,z,col_name):
    
    '''
    Returns the CONCETATED element value of 'm' indexed by its respective
        repeating field, 'field', and 'ind'.
    Updates the dataframe 'df' cell value indexed by 'z' and 'col_name'
    
    Parameters
    ----------
    field: list (with one element), required, for non-empty return choose valid 3 letter HL7 field
    ind: list, required, complete index path (as list) to desired element
    m: hl7 type object, required, m = hl7.parse(some_message)
    df:  pandas DataFrame, required
    z:  int, required, valid integer row index of df
    col_name: str, required, valid column in df
    
    Returns
    -------
    Str
        Concetated_Element separated by '|'
        
    Output
    ------
    Updates dataframe
        df.loc[z,col_name] = Concetated_Element
     
    Requirements
    ------------
    -Index_pull_CONC from pj_funcs.py
    -Pandas
    -hl7
    
    '''
    
    # Call the index on the message.
    obj = Index_pull_CONC(field,ind,m)
    
    # See if the 'obj' is an actual non-zero thing.
    if len(obj)>0:
        
        # If so, append to the row_z, col_colname in Dataframe, df
        df.loc[z,col_name] = obj
        
    # Else:  Do nothing.
    
    # Return the object
    return obj

############################################################################################################

def NSSP_Element_Grabber(data,Timed = True, Priority_only=False, outfile='None'):
    '''
    Creates dataframe of important elements from PHESS data.
    
    Parameters
    ----------
    data: pandas DataFrame, required, from PHESS sql pull
    
    Timed:  Default is True.  Prints total runtime at end.
    Priority_only:  Default is False.  
        If True, only gives priority 1 or 2 elements
    outfile:  Default is 'None':
        Replace with file name for dataframe to be wrote to as csv
        DO NOT INCLUDE .csv IF YOU CHOOSE TO MAKE ONE
    
    Returns
    -------
    dataframe
        
    Requirements
    ------------
    - import pandas as pd
    - import numpy as np
    - import time
    '''
    # Start our runtime clock.
    start_time = time.time()
    
    
    # Read in reader file as pandas dataframe
    reader = pd.read_excel('../data/processed/NSSP_Element_Reader.xlsx')
    
    # Create empty dataframe with rows we want interpreted from reader file
    df = pd.DataFrame(columns=reader['Processed Column'])
    
    # Create a few extra columns straight from our data file
    df['MESSAGE'] = data['MESSAGE']
    df['FACILITY_NAME'] = data['FACILITY_NAME']
    df['PATIENT_VISIT_NUMBER'] = data['PATIENT_VISIT_NUMBER']
    df['PATIENT_MRN'] = data['PATIENT_MRN']

    # Create a subset of rows from our reader file.  Only ones to loop through.
    # Order by 'Group_Order' so that some run before others that rely on previous.
    reader_sub = reader[reader.Ignore == 0].sort_values('Group_Order')

    # Loop through all data rows
    for z in np.arange(0,len(data)):
        
        # Locate our message
        message = df['MESSAGE'][z]
        
        # Decipher using hl7 function
        m = hl7.parse(message)
        
        # For each row in our reader file subset
        for j in np.arange(0,len(reader_sub)):
            
            # Initialize object.  Don't want one recycled from last loop
            obj=''
            
            # Choose the row we will use from the reader file
            row = reader_sub.iloc[j]
            
            # Identify element name we're working with.  Also a column name in output dataframe
            col_name = str(row['Processed Column'])
            
            # Identify code from our reader file we use to find the element in the HL7 message
            subcode = row['Code']
            
            # Does executing this code (originally a string) cause an error?
            ### NOTE:  calling locals and globals allows you to access all home-grown functions
            if NoError(exec,subcode,globals(), locals()):
                
                # If no errors, execute the code.
                exec(subcode,globals(), locals())


    # Some values may be empty strings and we do not want to count them as filled
    df = df.replace('',np.nan)
                
    # End time stopwatch
    end_time = time.time()

    # Unless they did not want it, print runtime
    if Timed != False:
        print(end_time-start_time)
    
    # If they only want priority elements:
    if Priority_only==True:
        # left = all columns interpreted from reader file
        left = df.iloc[:,:-4] 
        # right = MESSAGE, FACNAME, PATIENT_VN, PATIENT_MRN
        right = df.iloc[:,-4:]
        # find all cols we want from reader file. Priority cols
        priority_cols = reader['Processed Column'][(reader['Priority'] == 1.0)|(reader['Priority'] == 2.0)]
        # Index our left set by these columns 
        col_cut = left.loc[:,priority_cols]
        # glue left indexed with right again
        df = col_cut.join(right)
        
    # If they want an output file...
    if outfile!='None':
        # Specify output path and add csv bit.
        outpath = '../data/processed/'+outfile+'.csv'
        # No index
        df.to_csv(outpath, index=False)
    
    # return the dataframe!
    return df

############################################################################################################

def priority_cols(df, priority='both', extras=None, drop_cols=None):
    '''
    Spits out priority columns from a dataframe.
    Priority can be 1,2, or both.
    Extras indicate additional columns from the original dataframe you would like the output to contain.
    Drop_Cols indicate columns that you want to NOT include

    Parameters
    ----------
    df: pandas dataframe, required
    *priority: str, optional (default is both)
            'both' - returns priority 1 and priority 2 element columns
            'one' or '1' - returns priority 1 element columns only
            'two' or '2' - returns priority 2 element columns only
    *extras:  list, optional (default is None)
            list must contain valid column values from df.
    *drop_cols:  list, optional (default is None)
            list must contain valid column values from df.

    Returns
    -------
    pandas Dataframe
       
    Requirements
    ------------
    -import pandas as pd
    '''
    reader = pd.read_excel('../data/processed/NSSP_Element_Reader.xlsx') 
   
    # There is a chance that the user removed priority columns from the input dataframe for their own reasons.
    #     Therefore we only want to look at cases where the input dataframe columns match the reader processed column.
    reader = reader[reader['Processed Column'].isin(df.columns)]

    if priority.upper() == 'BOTH':
        cols = reader['Processed Column'][((reader.Priority == 1.0)|(reader.Priority == 2.0))]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
        
    elif (priority.upper() == 'ONE')|(priority == '1'):
        cols = reader['Processed Column'][(reader.Priority == 1.0)]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
    
    elif (priority.upper() == 'TWO')|(priority == '2'):
        cols = reader['Processed Column'][(reader.Priority == 2.0)]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
    
    else:
        print('Incorrect entry for specify.  Choose one of the following:  [\'both\',\'1\',\'2\']')

############################################################################################################


############################################################################################################

def validity_check(df, Timed=True):
    
    '''
    Checks to see which elements in a dataframe's specific NSSP priority columns meet NSSP validity standards.
    Returns a True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function    
    Timed - optional, boolean (True/False), default is True.  Returns time in seconds of completion.
    
    Returns
    --------
    validity_report - True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
    Requirements
    -------------
    import numpy as np
    import pandas as pd
    import time
    
    '''
    
    # Initialize Time
    start_time = time.time()
    
    
    # Read in the validity key
    key = pd.read_excel('../data/processed/NSSP_Validity_Reader.xlsx')

    # There is a chance that the user had decided to get rid of columns we typically check for validity.
    #      Therefore we need to only loop through 'key' rows that match our input dataframe's columns.
    key = key[key['Element'].isin(df.columns)]
    
    # Initialize empty pandas dataframe
    validity_report = pd.DataFrame()
    
    # Make sure we know nan means NaN
    nan = np.nan
    
    # Loop through each row in our validity key file
    for i in np.arange(0,len(key)):
        
        # Locate the row that our loop is on.  Define:
        row = key.iloc[i]
        
        # The element name 
        col_name = row['Element']
        
        #######################################################################################################
        #  All NSSP Priority Elements have validity checks that fall into one of the following 4 categories.
        #######################################################################################################

        # The list that the value may need to be part of to be valid.
        row_list = row['List']
        
        # The list that the value should not be part of to be valid.
        row_notlist = row['NOT_List']
        
        # The upper bound of a numeric value that it needs in order to be valid.
        row_bounds = row['Bounds']
        
        # The string fomat (in RegEx format) that a value needs to be valid.
        row_format = row['Format']

        #######################################################################################################
        # Check to see if this element has a non-null entry for one of the 4 broad criteria.
        #     If it does (which will only work for one of the four):
        #           Execute the code on the Element's who data column.
        #           Append a newly formed True/False array as a column to our output validity report
        #######################################################################################################
        
        if (row_list == row_list):
            listy = row_list.split(',')
            validity_report[col_name] = (df[col_name].str.upper().isin(listy))

        elif (row_notlist == row_notlist):
            nonlist = row_notlist.split(',')
            validity_report[col_name] = (~df[col_name].str.upper().isin(nonlist))  

        elif (row_bounds == row_bounds):
            num = 120
            validity_report[col_name] = pd.to_numeric(df[col_name]) < num

        elif (row_format == row_format):
            search = row_format
            validity_report[col_name] = (df[col_name].str.contains(search,na=False))
            
            
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
            
    return validity_report

############################################################################################################

def Visualize_Facility_DQ(df, fac_name, hide_yticks = False, Timed = True):
    '''
    Returns Visualization of data quality in the form of a heatmap.
    Rows are all individual visits for the inputted facility.
    Columns are NSSP Priority elements that can be checked for validity.
    Color shows valid entries (green), invalid entries (yellow), and absent entries (red)
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function
    fac_name - required, str, valid name of facility.
        if unsure of valid entry options, use the following code for options:
        df['FACILITY_NAME'].unique()   # may need to change for your df name
    
    Returns
    --------
    out[0] = Pandas dataframe used to create visualization.  2D composed of 0s (red), 1s (yellow), 2s (green)
    out[1] = Pandas dataframe of data behind visit.  Multiple HL7 messages composing 1 visit concatenated by '~' character
    
    Output
    -------
    sns.heatmap visualization
    
    Requirements
    -------------
    import numpy as np
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    
    '''

    # Initialize Time
    start_time = time.time()
    
    # Create sub-dataframe of only visits within a facility
    hosp_visits = df[df.FACILITY_NAME==fac_name]

    # Read in our validity key
    key = pd.read_excel('../data/processed/NSSP_Validity_Reader.xlsx')

    # There is a chance that the user had decided to get rid of columns we typically check for validity.
    #      Therefore we need to only loop through 'key' rows that match our input dataframe's columns.
    key = key[key['Element'].isin(df.columns)]

    # Initialize data quality array (0s 1s 2s)
    out0 = pd.DataFrame(columns=key.Element)
    
    # Initialize data represented array (hl7 data concatenated by ~ character)
    out1 = pd.DataFrame(columns=key.Element)

    # Set original index to 0.  Will increase by 1 after every visit has its info captured.
    cur_index = 0

    # Group by MRN
    MRN_group = hosp_visits.groupby('PATIENT_MRN')

    # Loop through our MRN Groups
    for index,frame in MRN_group:

        # Group by Visit Number
        VISIT_group = frame.groupby('PATIENT_VISIT_NUMBER')

        # Loop through VISITS
        for index2,frame2 in VISIT_group:

            # Initialize dataframe
            one_visit = pd.DataFrame()

            # Only look at visit info that can be checked for validity (must be a validity key element)
            impz = frame2.loc[:,key.Element]

            # Create correct format input for validity check.
            #    needs FACILITY_NAME,PATIENT_VISIT_NUMBER,PATIENT_MRN
            impz2 = impz.copy()
            impz2['FACILITY_NAME'] = fac_name
            impz2['PATIENT_MRN'] = index
            impz2['PATIENT_VISIT_NUMBER'] = index2

            # Run a validity check on our visit's important columns
            one_visit = validity_check(impz2,Timed=False).iloc[:,:]

            # Completness returns 1D list of 0s / 1s determining if there is a non-null value in each column
            completeness = ((~impz.isnull()).sum() != 0).astype(int)

            # Validness returns 1D list of 0s / 1s determining if there is a valid value in each column
            validness = (one_visit.sum() != 0).astype(int)

            # Sum completness + validness to get picture for overall data quality
            tots = completeness+validness

            # Save this overall data quality score into our out0 array and save the index (Patient Visit Number)
            out0.loc[cur_index,:] = tots
            out0.loc[cur_index,'PATIENT_VISIT_NUMBER'] = index2

            # Also save our data that has been assessed for quality.  Concatenate by '~' character
            # First replace all NaN with empty character.  Need this to concat strings together.
            impz_no_na = impz.fillna('')
            
            for col in impz_no_na.columns:
                newcol = '~'.join(impz_no_na[col].astype(str))
                out1.loc[cur_index,col] = newcol

        
            # Visit over, onto the next.  Increase current index by +1
            cur_index += 1

    # Reset our arbitrary 0-n index and replace with the patient visit number
    out0.reset_index()
    out0.set_index('PATIENT_VISIT_NUMBER')

    # Look at how many visits we have
    num_visits = len(out0)

    # Create scalar (just made sense in my head) for figure scaling.
    scalar = int(num_visits/20)+1
    
    # Create figure/axes with my respective scaling
    fig, ax = plt.subplots(figsize=(20/scalar,1*num_visits/scalar))

    # Create custom colormap
    my_cmap = colors.ListedColormap(['Red','Yellow','Green'])

    # Make a heatmap of our 0,1,2 array of absent, invalid, valid elements.
    #     specify linewidth, xticks, yticks, linecolor separation to black
    heatmap = sns.heatmap(np.array(out0)[:,:-1].astype(int),cmap=my_cmap,linewidth=0.5,
                          xticklabels=key.Element, linecolor='k',center=1,
                         yticklabels=out0.PATIENT_VISIT_NUMBER)
    
    # Hide yticks if necissary
    if hide_yticks == True:
        plt.yticks([])


    ###################################################
    # Plot customization
    ###################################################

    # Increase size of xticks and x/y axes
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15) 
    #matplotlib.rc('axes', labelsize=25) 
    plt.rc('axes', titlesize=25)     
    plt.rc('axes', labelsize=20)
    #matplotlib.rc('title', labelsize=30) 

    # Set colorbar axis
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.33, 1, 1.66])
    cbar.set_ticklabels(['Absent', 'Invalid', 'Valid'])

    # Set Title
    plt.title('NSSP Priority Element\nData Visualization\n'+fac_name)    
    
    # Set and rotate xticks 90 deg
    plt.xticks(rotation=90) 
    plt.xlabel('NSSP Element')

    # Set ylabel
    plt.ylabel('Patient Visit Number')

    # Show your result
    plt.show()
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    return out0,out1

####################################################################################################################################


def issues_in_messages(df, Timed=True, combine_issues_on_message = False, split_issue_column = False):
    '''
    Processes dataframe outputted by NSSP_Element_Grabber() function.
    Outputs dataframe describing message errors.  See optional args for output dataframe customation.
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function
    *Timed - optional, bool, default is True.  Outputs runtime in seconds upon completion.
    *combine_issues_on_message - optional, bool, default is False.  SEE (2) below
    *split_issue_column - optional, bool, default is False.  SEE (3) below
    
    
    NOTE:  only one of 'combine_issues_on_message' or 'split_issue_column' can be True
    
    Returns
    ----------------------------------------------------------------------------
    Pandas dataframe. Columns include:
    
    (1)
    DEFAULT: WHEN split_issue_colum = False , combine_issue_on_message = False
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    Issue -> string concatenation of 'error_type|element_name|priority|description|valid_options|message_value|suggestion|comment'
    
    ------
    
    (2)
    WHEN combine_issue_on_message = True, split_issue_colum = False 
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    Issue -> string concatenation of 'error_type|element_name|priority|description|valid_options|message_value|suggestion|comment'
             MULTIPLE string concatenations per cell, separated by newline '\n'
    
    Num_Missings -> number of issues that had a type of 'Missing or Null'
    Num_Invalids -> number of issues that had a type of 'Invalid'
    Num_Issues_Total -> number of total issues
    
    ------
    
    (3)
    WHEN combine_issue_on_message = False , split_issue_colum = True
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    error_type -> 'Missing or Null' or 'Invalid'
    element_name -> NSSP Priority Element name with issue
    priority -> NSSP Priority '1' or '2'
    description -> Describes location/parameters of element in HL7 message
    valid_options -> IF element can be checked for validity, describes a valid entry.
    message_value -> IF element was determined as invalid, give the invalid element value.
    suggestion -> IF element was determined as invalid, give an educated guess as to what they meant.
    comment -> IF element was determined as invalid, give feedback/advice on the message error.
    
    
    --------------------------------------------------------------------------------
    
    Requirements
    -------------
    from pj_funcs import *
    import numpy as np
    import pandas as pd
    import time
    
    '''

    if (combine_issues_on_message==True)&(split_issue_column==True):
        print('ERROR:  Only 1 of: combine_issues_on_message / split_issue_column can be True ')
        return -1
    
    # Initialize Time
    start_time = time.time()
    ########################################################################################################
    # Create dataframe of 0s, 0.5s, 1s representing missing/null , invalid, and valid values.
    ########################################################################################################


    # we only want to look at priority columns
    new = priority_cols(df)

    #  Create a new column combining all information to group by visit
    new['Grouper_ID'] = df.FACILITY_NAME+'|'+df.PATIENT_MRN+'|'+df.PATIENT_VISIT_NUMBER

    ############

    # Run a validity check on the priority columns
    vc = validity_check(new,Timed=False)

    # Validity check only outputs priority columns so redefine our grouper ID.  We will set this to be our index
    vc['Grouper_ID'] = new['Grouper_ID']
    vc = vc.set_index('Grouper_ID')

    # Create a copy of our dataframes priority cols (new) and call it df1.  Set its index
    df1 = new.copy()
    df1 = df1.set_index('Grouper_ID')

    ############

    # Ones that have a non-empty value we will asign a value of 1 to.  Null-values will be assigned 0
    df_comp = (~df1.isnull()).astype(int) 

    # Validity check will also be interpreted as an integer.  
    df_vc = vc.astype(int)

    ############

    # Invalid entries will now be represented as -0.5 in our validity df
    df_vc = df_vc.replace(0,-0.5)

    # Valid entries will be represented as 0 in our validity df
    df_vc = df_vc.replace(1,0)

    ############

    # Find columns that can be checked for validity
    c = df_comp.columns.intersection(df_vc.columns)

    # For these columns, we want to sum our two dataframes [df_vc + df_comp]
    df_comp[c] =  df_comp[c].add(df_vc[c], fill_value=0)

    # NOTE at this point df_comp is an array of -0.5s, 0s, 1s, 2s. If a value that could be invalid was empty, the sum was 0+(-0.5)

    # Replace -0.5 with 0.  Represents an empty visit regarless of if it's also invalid
    df_comp = df_comp.replace(-0.5,0)

    # Reset the index and make a new, copied MESSAGE column
    df_comp = df_comp.reset_index()
    df_comp['MESSAGE'] = df['MESSAGE']

    ######################################################################

    # set the index of new again
    new  = new.set_index('Grouper_ID')

    # Replace any instance of | to ~ because we will later use pipe characters for an important purpose
    new = new.replace('\|','~', regex=True)
    new = new.reset_index()

    ########################################################################################################
    # Begin our part where we Create the dataframe 
    ########################################################################################################

    # Load our key and set its index
    key = pd.read_excel('../data/processed/Message_Corrector_Key.xlsx')
    key = key.set_index('Element')

    # Initialize df_out
    df_out = pd.DataFrame(columns=['MESSAGE','Grouper_ID','Issue'])
    cur_index = 0

    # To save time on efficiency, we write our rows directly instead of appending them (which rewrites array)
    #     to write rows directly, you need a rough estimate of how many rows you have.  We just did it exactly
    pointless = np.array([''] * (((df_comp == 0)|(df_comp == 0.5)).sum().sum()))
    df_out['DELETE_L8R'] = pointless


    # Loop through all rows in our dataframe of 0s,0.5s,1s (called df_comp)
    for i in np.arange(0,len(df_comp)):

        # Each row will have a grouper and Message we will eventually store in our output dataframe
        grouperID = df_comp['Grouper_ID'].iloc[i]
        message = df_comp['MESSAGE'].iloc[i]

        # Loop through all of the columns in our row
        for col in df_comp.columns:

            # Initialize empty list.  We will fill with strings and concatenate to fill our 'Issue' column if missing/invalid
            entry = []

            # See if the current cell value is 0 -> representing null/missing value
            if df_comp.loc[i,col] == 0:

                # Append the problem, the element name (col), the element priority, the element description
                entry.append('Missing or Null')
                entry.append(str(col))
                entry.append(str(key.loc[col,'Priority']))
                entry.append(str(key.loc[col,'Description']))

                # If there is a list of valid options in our key, append that, otherwise append empty string
                if (key.loc[col,'Valid_Options']) == (key.loc[col,'Valid_Options']):
                    entry.append(str(key.loc[col,'Valid_Options']))
                else:
                    entry.append('')

                # Append empty strings for Message entry, Comment, and Suggestion
                entry.append('')
                entry.append('')
                entry.append('')

            ########################################################################################################

            # See if the current cell value is 0.5 -> representing an invalid value
            elif df_comp.loc[i,col] == 0.5:

                # Append the problem, the element name (col), the element priority, the element description
                entry.append('Invalid')
                entry.append(str(col))
                entry.append(str(key.loc[col,'Priority']))
                entry.append(str(key.loc[col,'Description']))

                # Our df_comp cell was 0.5, therefore there is a list of valid options in our key, append that
                entry.append(str(key.loc[col,'Valid_Options']))

                # Append the value that was determined to be invalid.  DataFrame called new contains all initial cell values.
                entry.append(str(new.loc[i,col]))

                # Initialize our comment and suggestion.  If comment/suggestion exists, our executed code will replace these
                comment = ''
                suggestion = ''

                # See if we have a non-null code value.  
                if (key.loc[col,'Suggestion_Code']) == (key.loc[col,'Suggestion_Code']):

                    # Nearly all of these will call on our invalid value.  Define that
                    invalid_value = str(new.loc[i,col])

                    # Execute the code within the cell. Exec will append comment/suggestion to entry
                    code_to_run = str(key.loc[col,'Suggestion_Code'])
                    exec(code_to_run,globals(),locals())
                
                # If we don't have any code to exec(ute), append empty comment/suggestion to issue string
                else:
                    entry.append(comment)
                    entry.append(suggestion)




            ########################################################################################################

            # If there was a problem (either missing/invalid) the list called entry will be non-empty
            if len(entry) > 0:

                # Join our entries by a pipe character
                issue_string = '|'.join(entry)

                # Append our 3 column row to our output dataframe at the current index
                df_out.loc[cur_index] = [message,grouperID,issue_string,'']

                # Update the current index
                cur_index += 1


    # Delete the axis we initially made just to set length
    df_out = df_out.drop('DELETE_L8R',axis=1)

    
    ##############################################
    # Optional Args TIME
    ##############################################
    
    if (combine_issues_on_message == True):
    
        # create empty dataframe.  Correct length (column-wise).
        comb_on_issue = pd.DataFrame(columns=['MESSAGE','Grouper_ID','Issue'])

        # create the correct lengthed (row-wise) dataframe by recognizing all unique messages.  
        comb_on_issue.MESSAGE = df_out.MESSAGE.unique()

        # initialize a count
        count = 0

        # Loop through groupby objects when we group by MESSAGE
        for index,frame in df_out.groupby('MESSAGE'):

            # Identify the message (which is the index) and the grouper_ID (same for all parts of frame. Arbitrarily choose first index)
            message = index
            grouperID = frame.Grouper_ID.iloc[0]

            # Drop all duplicate rows.  Some messages may appear more than once in our original dataset
            frame2 = frame.drop_duplicates()

            # Combine all unique Issue values by a newline seperator.
            comb_issue = frame2.Issue.str.cat(sep='\n')   

            # Append our new info to dataframe and update our count.  
            comb_on_issue.iloc[count] = [message,grouperID,comb_issue]  
            count+=1
            
        # Create some extra columns describing number of types of errors
        comb_on_issue['Num_Missings'] = comb_on_issue.Issue.str.count('Missing or Null')
        comb_on_issue['Num_Invalids'] = comb_on_issue.Issue.str.count('Invalid\|')
        comb_on_issue['Num_Issues_Total'] = comb_on_issue['Num_Missings'] + comb_on_issue['Num_Invalids']
        
        # Rename our df_out so that we can only return one thing
        df_out = comb_on_issue

    ######################################################################################################################
    
    if (split_issue_column == True):
        expanded_issue = df_out.Issue.str.split('\|',expand=True)
        expanded_issue.columns = ['Issue_Type','Element_Name','Priority','Description','Valid_Options',
                                  'Message_Value','Suggestion','Comment']
        
        # Rename our df_out so that we only return one thing
        df_out = df_out[['MESSAGE','Grouper_ID']].join(expanded_issue)
    
    ######################################################################################################################

    # Keep track of end time
    end_time = time.time()

    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
        
    return df_out
