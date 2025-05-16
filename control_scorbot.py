import matlab.engine as mlab
import time

#palm up gesture

eng = mlab.start_matlab()

functions = {
    'palm_up': 'eng.Handover_block(nargout = 0)',
    'Okay': eng.Align_three_blocks(nargout= 0),
    'peace_sign': eng.Stack3blocks(nargout= 0),
    'B-sign': eng.closegripper(nargout= 0),
    'A-sign': eng.opengripper(nargout = 0),
    'thumbs_down': eng.shutdown(nargout= 0),
    'thumbs_up': eng.initialize(nargout= 0),
    'C-sign': eng.Home(nargout= 0)
}
def execute_scorbot_command(sign):
    sign = str(sign)
    eng = mlab.start_matlab()
    functions[sign]
    eng.quit()


execute_scorbot_command('palm_up')






