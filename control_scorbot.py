import matlab.engine as mlab
import time

# Start MATLAB engine
eng = mlab.start_matlab()
eng.Handover_block(nargout = 0)
time.sleep(5)



eng.GoHome(nargout = 0)

eng.quit()








