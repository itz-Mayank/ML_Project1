# To handel the Exceptions, we will use this exception.py file.

import sys
# The sys library in Python is used to access system-specific parameters and functions, such as reading command-line arguments, managing the Python runtime environment, and controlling program execution.
from src.logger import logging

def error_messege_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = 'Error occured in python script name [{0}] line number [{1}] error messege[{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    
    return error_message
    
    
class CustomException(Exception):
    def __init__(self, error_messege,error_detail:sys):
        super().__init__(error_messege)
        self.error_messege = error_messege_detail(error_messege,error_detail=error_detail)
        
    def __str__(self):
        return self.error_messege
    
# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info('One is divided by Zero.')
#         raise CustomException(e,sys)
    