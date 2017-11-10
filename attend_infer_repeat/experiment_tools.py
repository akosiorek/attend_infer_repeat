import imp
import importlib
import os
import sys
import re
import shutil
import json
import subprocess
import tensorflow as tf


FLAG_FILE = 'flags.json'

# TODO: docs


def json_store(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


def init_checkpoint(checkpoint_dir, data_config, model_config, restart):
    """
    1) try mk checkpoint_dir
    2) if continue:
        a) check if checkpoint_dir/n where n is an integer and raise if it doesnt
        b) load flags
        c) load checkpoint

    3) if not:
        a) n=n+1, mkdir
        b) store flags
        c) copy data & model configs

    :param checkpoint_dir:
    :param data_config:
    :param model_config:
    :return:
    """

    # check if the experiment folder exists and create if not
    checkpoint_dir_exists = os.path.exists(checkpoint_dir)
    if not checkpoint_dir_exists:
        if restart:
            raise ValueError("Can't restart when the checkpoint dir '{}' doesn't exist.".format(checkpoint_dir))
        else:
            os.makedirs(checkpoint_dir)

    elif not os.path.isdir(checkpoint_dir):
        raise ValueError("Checkpoint dir '{}' is not a directory.".format(checkpoint_dir))

    experiment_folders = [f for f in os.listdir(checkpoint_dir) if not f.startswith('_')]
    if experiment_folders:
        experiment_folder = int(sorted(experiment_folders, key=lambda x: int(x))[-1])
        if not restart:
            experiment_folder += 1
    else:
        if restart:
            raise ValueError("Can't restart since no experiments were run before in checkpoint dir '{}'.".format(checkpoint_dir))
        else:
            experiment_folder = 1

    experiment_folder = os.path.join(checkpoint_dir, str(experiment_folder))
    if not restart:
        os.mkdir(experiment_folder)

    flag_path = os.path.join(experiment_folder, FLAG_FILE)
    restart_checkpoint = None
    if restart:
        flags = json_load(flag_path)
        _restore_flags(flags)
        model_files = find_model_files(experiment_folder)
        if model_files:
            restart_checkpoint = model_files[max(model_files.keys())]

    else:
        # store flags
        _load_flags(model_config, data_config)
        flags = parse_flags()
        assert_all_flags_parsed()

        try:
            flags['git_commit'] = get_git_revision_hash()
        except subprocess.CalledProcessError:
            # not in repo
            pass

        json_store(flag_path, flags)

        # store configs
        for src in (model_config, data_config):
            file_name = os.path.basename(src)
            dst = os.path.join(experiment_folder, file_name)
            shutil.copy(src, dst)

    return experiment_folder, flags, restart_checkpoint


def extract_itr_from_modelfile(model_path):
    return int(model_path.split('-')[-1].split('.')[0])


def find_model_files(model_dir):
    pattern = re.compile(r'.ckpt-[0-9]+$')
    model_files = [f.replace('.index', '') for f in os.listdir(model_dir)]
    model_files = [f for f in model_files if pattern.search(f)]
    model_files = {extract_itr_from_modelfile(f): os.path.join(model_dir, f) for f in model_files}
    return model_files


def load(conf_path, *args, **kwargs):

    module, name = _import_module(conf_path)
    try:
        load_func = module.load
    except AttributeError:
        raise ValueError("The config file should specify 'load' function but no such function was "
                           "found in {}".format(module.__file__))

    print "Loading '{}' from {}".format(module.__name__, module.__file__)
    return load_func(*args, **kwargs)


def _import_module(module_path_or_name):
    module, name = None, None

    if module_path_or_name.endswith('.py'):
        file_name = module_path_or_name
        module_path_or_name = os.path.basename(os.path.splitext(module_path_or_name)[0])
        if module_path_or_name in sys.modules:
            module = sys.modules[module_path_or_name]
        else:
            module = imp.load_source(module_path_or_name, file_name)
    else:
        module = importlib.import_module(module_path_or_name)

    if module:
        name = module_path_or_name.split('.')[-1].split('/')[-1]

    return module, name


def _load_flags(*config_paths):
    """Aggregates gflags from `config_path` into global flags

    :param config_paths:
    :return:
    """
    for config_path in config_paths:
        print 'loading flags from', config_path
        _import_module(config_path)


def parse_flags():
    f = tf.flags.FLAGS
    args = sys.argv[1:]

    old_flags = f.__dict__['__flags'].copy()
    # Parse the known flags from that list, or from the command
    # line otherwise.
    # pylint: disable=protected-access
    flags_passthrough = f._parse_flags(args=args)
    sys.argv[1:] = flags_passthrough
    f.__dict__['__flags'].update(old_flags)

    # pylint: disable=protected-access
    return f.__flags


def _restore_flags(flags):
    # TODO: this should still parse cli flags and use them in case of
    # restarting a job from a new commit where flags where added.
    # Right now it results in a runtime error because a flag might be request that hasn't been defined
    # at the time of the first run.
    tf.flags.FLAGS.__dict__['__flags'] = flags
    tf.flags.FLAGS.__dict__['__parsed'] = True


def print_flags():
    flags = tf.flags.FLAGS.__flags

    print 'Flags:'
    keys = sorted(flags.keys())
    for k in keys:
        print '{}: {}'.format(k, flags[k])


def set_flags(**flag_dict):
    for k, v in flag_dict.iteritems():
       sys.argv.append('--{}={}'.format(k, v))


def assert_all_flags_parsed():
    not_parsed = [a for a in sys.argv[1:] if a.startswith('--')]
    if not_parsed:
        raise RuntimeError('Failed to parse following flags: {}'.format(not_parsed))


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def set_flags_if_notebook(**flags_to_set):
    if is_notebook() and flags_to_set:
        print 'Setting the following flags:'
        for k, v in flags_to_set.iteritems():
            print '\t{}: {}'.format(k, v)

        set_flags(**flags_to_set)


def is_notebook():
    notebook = False
    try:
        interpreter = get_ipython().__class__.__name__
        if interpreter == 'ZMQInteractiveShell':
            notebook = True
        elif interpreter != 'TerminalInteractiveShell':
            raise ValueError('Unknown interpreter name: {}'.format(interpreter))

    except NameError:
        # get_ipython is undefined => no notebook
        pass
    return notebook


if __name__ == '__main__':

    import tensorflow as tf
    flags = tf.flags

    tf.flags.DEFINE_integer('int_flag', -2, 'some int')
    tf.flags.DEFINE_string('string_flag', 'abc', 'some string')

    checkpoint_dir = '../checkpoints/setup'
    data_config = 'configs/static_mnist_data.py'
    model_config = 'configs/imp_weighted_nvil.py'


    # sys.argv.append('--int_flag=100')
    # sys.argv.append('--model_flag=-1')
    # print sys.argv

    experiment_folder, loaded_flags, checkpoint_dir = init_checkpoint(checkpoint_dir, data_config, model_config, restart=False)

    print experiment_folder
    print loaded_flags
    print checkpoint_dir
    print sys.argv

    print
    print 'tf.flags:'
    for k, v in tf.flags.FLAGS.__flags.iteritems():
        print k, v
    # batch_size = 64
    # data_dict = load(data_config, batch_size)
    # print data_dict.keys()
    #
    # air, train_step, global_step = load(model_config, img=data_dict.train_img, num=data_dict.train_num)
    #
    # print air

