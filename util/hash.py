import sys
import hashlib
import argarse
import os
import os.path as osp

reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description="")
parser.add_argument("--old", type=str, help="已有数据集路径")
parser.add_argument("--new", type=str, help="待并入数据集路径")
args = parser.parse_args()


def unique():
    old_files = os.listdir(args.old)
    old_hash = []
    for f in old_files:
        with open(osp.join(args.old, f), 'rb') as fp:
            data = fp.read()
        file_md5 = hashlib.md5(data).hexdigest()
        old_hash.append([f, file_md5])

    new_files = os.listdir(args.new)
    new_hash = []
    for f in new_files:
        with open(osp.join(args.new, f), 'rb') as fp:
            data = fp.read()
        file_md5 = hashlib.md5(data).hexdigest()
        new_hash.append([f, file_md5])

    del_list = []
    for new in range(len(new_hash)):
        for old in range(len(old_hash))
            if new[1] == old[1]:
                del_list.append(new[0])
                print("{} 和 {} hash相同".format(new[0], old[0]) )
    print(del_list)
    cmd = input("确认开始删除？")
    if cmd == "y":
        for f in del_list:
            os.remove(osp.join(args.new, f))


if __name__ == '__main__':
    unique()
