import matlab.engine as mlab
import time

def execute_matlab_commands():
    eng = mlab.start_matlab()
    eng.Handover_block(nargout=0)
    time.sleep(5)
    eng.GoHome(nargout=0)
    eng.quit()








