import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ann", type=str, help="标签路径")
parser.add_argument("--curr", type=str, help="当前标签中类别")
parser.add_argument("--after", type=str, help="目标标签类别")
args = parser.parse_args()

for name in os.listdir(args.ann):
    path = osp.join(args.ann, name)
    with open(path, "r") as f:
        content = f.read()
    content = content.replace(args.curr, args.after)
    with open(path, "w") as f:
        print(content, file=f)
