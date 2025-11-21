'''
More common ml tools
'''

# %%
import os
import sys
import subprocess
import time


# %%
def get_run_info(args):
    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    args.datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    args.user = os.getenv('USER')
    args.sys_argv = sys.argv
    args.script = os.path.abspath(__file__)
    return args
