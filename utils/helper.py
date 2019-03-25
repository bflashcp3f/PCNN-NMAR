import os
import operator

def check_dir(dir, verbose=True):
    if not os.path.exists(dir):
        if verbose:
            print "Directory %s do not exist; creating..." % dir
        os.makedirs(dir)
        
        
def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in sorted(config.items(), key=operator.itemgetter(0)):
        info += "\t%s : %s\n" % (k, str(v))
    print "\n" + info + "\n"
    return


def sort_bags_by_size(train_data):
    """
    To have a more stable result
    """
    key_len = [(key, len(bag)) for key, bag in train_data.items()]
    sorted_key_len = sorted(key_len, key=operator.itemgetter(1), reverse=True)
    
    return [item[0] for item in sorted_key_len]
    
    