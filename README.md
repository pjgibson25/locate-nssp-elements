# README:  Locate NSSP Elements

## Background

I was given the task of creating Python scripts to parse NSSP Priority Elements out of an ADT HL7 Message.
See terminology definitions below:

* HL7 Messages - A tree-type data structure that contains information related to a hospital visit
* ADT - Acronym for Admission/Discharge/Transfer records.
* NSSP Priority Elements - Important information from the visit.
Importance determined by National Syndromic Surveillance Program (NSSP). 
Examples of NSSP Priority elements include Patient_Age and Admit_Date_Time

It should be noted that for many messages, some NSSP Priority Elements are missing. 
This could be a result of problematic hospital procedures, a patient not disclosing information, or complicated visits.
Some elements are located in one message location and are relatively easy to parse.
Other elements follow heirchical arguments (ex:  Take the first non-null value of location1, location2).
More complex elements require even more complex if-elif-else logic.

For a list of all NSSP Priority Elements and their descriptions, see the excel file <b>'data/processed/NSSP_Element_Reader.xlsx'</b>

-------------

Once I was able to pull NSSP Priority elements from an HL7 message, I moved on to assessing the completness and timeliness of the HL7 Messages in accordance with NSSP standards.

The last project task was to create a comprehensive dataframe of message errors.
In general, each row describes the error and how to fix it.

## Project Structure

There are three primary folders and one optional folder within this repository:
* data - where data files are stored
  * raw/ - <b><u>ACTION REQUIRED</u></b> directory where the user (you) must put SQL-outputted csv dataframes from PHESS-ED data
  * processed/ - directory where you can save output dataframes to.
    * NSSP_Element_Reader - file describing element locations and code to execute in our NSSP_Element_Grabber() function.
    * NSSP_Validity_Reader - file describing characteristics of valid NSSP Priority Elements and code as to check for validity.
    * Message_Corrector_Key - file with important information about element locations and code to provide suggestions for invalid values.
    
* python 
  *dakf





Because we need to parse through an HL7 message, a complex tree structured dataframe that varies message-to-message, I use the help of the [HL7 Library](https://python-hl7.readthedocs.io/en/latest/).
The library is mostly useful for indexing an HL7 Message.

