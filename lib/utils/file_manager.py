import cPickle as pickle


def pickleSave(filename, obj):
    f = open(filename, 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickleLoad(filename):
    f = open(filename, 'rb')
    return pickle.load(f)
