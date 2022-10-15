import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'meta_data')


def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list=None, array_command=None, array_indices=None, n_cpus=1, n_gpus=1, dry=False,
                          mem=3000, length="short",
                          mode='local', gpu_model=None, prompt=True, gpu_mtotal=None,
                          relaunch=False, relaunch_after=None, output_filename="", output_path_prefix=""):

    # check if single or array
    is_array_job = array_command is not None and array_indices is not None
    assert (command_list is not None) or is_array_job
    if is_array_job:
        assert all([(ind.isdigit() if type(ind) != int else True) for ind in array_indices]), f"array indices must be positive ints but got `{array_indices}`"

    if mode == 'local':
        if prompt and not dry:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if is_array_job:
            command_list = [array_command.replace("\$LSB_JOBINDEX", str(ind)) for ind in array_indices]

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd, end="\n\n")
                else:
                    subprocess.call(cmd, shell=True)

    else:
        raise NotImplementedError
